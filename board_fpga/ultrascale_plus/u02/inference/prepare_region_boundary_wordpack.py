"""Remove the evidenced quantized-RAM backedge in an independent candidate."""
import argparse
import json
from pathlib import Path
import time
from prepare_region_boundary_pipeline import U02, sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation',type=Path,required=True)
    args=parser.parse_args()
    numeric_path=args.validation/'result.json';numeric=json.loads(numeric_path.read_text())
    if not numeric.get('complete') or numeric.get('real_boundaries')!=70:
        raise ValueError('Validated pipeline boundary preparation required')
    for name,digest in numeric['input_sha256'].items():
        if sha(Path(name))!=digest:raise ValueError('Validated input changed')
    parent=Path(numeric['preparation']);plan=json.loads((parent/'preparation.json').read_text())
    record=U02/('reports/region_boundary_wordpack_preparation_'+str(time.time_ns()));record.mkdir()
    additions=[
        ('template<class Destination> inline int region_prepare_boundary_words_to(',
         'inline int region_prepare_boundary_words_to('),
        ('        Destination quantized,size_t quantized_bytes)',
         '        RegionWordDestination quantized,size_t quantized_bytes)'),
        ('    for(int c=0;c<first.ci;++c)for(size_t p=0;p<plane;++p) {\n',
         '    uint64_t packed_word=0;\n    for(int c=0;c<first.ci;++c)for(size_t p=0;p<plane;++p) {\n'),
        ('        quantized.set(size_t(c)*plane+p,int8_t(value<0?-rounded:rounded));',
         '        const size_t index=size_t(c)*plane+p;\n'
         '        const unsigned shift=unsigned(index%8)*8;\n'
         '        if(!shift)packed_word=0;\n'
         '        packed_word|=uint64_t(uint8_t(int8_t(value<0?-rounded:rounded)))<<shift;\n'
         '        if(shift==56 || index+1==elements)quantized.data[index/8]=packed_word;')]
    files={}
    for name,entry in plan['files'].items():
        source=parent/name
        if sha(source)!=entry['candidate_sha256']:raise ValueError('Parent header changed')
        text=source.read_text()
        if name=='region_boundary_words.h':
            for before,after in additions:
                if text.count(before)!=1:raise ValueError('Ambiguous wordpack transform')
                text=text.replace(before,after)
        target=record/name;target.write_text(text)
        files[name]=dict(baseline_sha256=entry['baseline_sha256'],candidate_sha256=sha(target))
    result=dict(plan)
    result.update(complete=False,numerics_verified=False,files=files,
                  parent_validation=str(args.validation.resolve()),parent_validation_sha256=sha(numeric_path),
                  preparation_script=str(Path(__file__).resolve()),preparation_script_sha256=sha(Path(__file__)),
                  changes=plan['changes']+[dict(before=a,after=b) for a,b in additions],
                  scope='Only replace per-byte quantized RAM read/modify/write with a local 64-bit accumulator and one write per eight bytes or tail. All arithmetic/coefficients, rejection timing and original RAM bindings retained; numerical and actual II verification required.')
    (record/'preparation.json').write_text(json.dumps(result,indent=2)+'\n')
    print('U02_REGION_BOUNDARY_WORDPACK_PREPARATION',record)


if __name__=='__main__':main()
