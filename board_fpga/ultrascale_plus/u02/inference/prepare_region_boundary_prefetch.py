"""Prepare contiguous FP32 prefetch into bounded private URAM before both scans."""
import argparse
import json
from pathlib import Path
import time
from prepare_region_boundary_pipeline import U02, sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path,required=True)
    args=parser.parse_args()
    parent_proof=args.parent/'result.json';proof=json.loads(parent_proof.read_text())
    if not proof.get('complete') or not proof.get('boundary_numerics_reused'):
        raise ValueError('Verified wordpack/URAM binding required')
    for path,digest in proof['input_sha256'].items():
        if sha(Path(path))!=digest:raise ValueError('Verified parent input changed: '+path)
    manifest=args.parent/'preparation.json';plan=json.loads(manifest.read_text())
    if not plan.get('weight_uram'):raise ValueError('Weight staging must retain separate URAM binding')
    before='    const size_t plane=size_t(first.h)*first.w;'
    after='''    // At most 98304 floats plus one leading half-word for an odd offset.
    // Fetch each AXI word once, then scan/quantize entirely from private URAM.
    uint64_t fp32_prefetch[49153];
#pragma HLS RESOURCE variable=fp32_prefetch core=RAM_2P_URAM
    const size_t local_offset=input_offset%2;
    const size_t input_words=(elements+local_offset+1)/2;
    if(input_words>49153)return -3;
    PREFETCH_FP32_WORDS: for(size_t i=0;i<input_words;++i) {
#pragma HLS PIPELINE II=1
        fp32_prefetch[i]=input[input_offset/2+i];
    }
    const size_t plane=size_t(first.h)*first.w;'''
    transforms=[(before,after),
                ('region_word_float(input,input_offset+(size_t(c)*plane+p))',
                 'region_word_float(fp32_prefetch,local_offset+(size_t(c)*plane+p))')]
    record=U02/('reports/region_boundary_prefetch_preparation_'+str(time.time_ns()));record.mkdir()
    files={}
    for name,entry in plan['files'].items():
        source=args.parent/name
        if sha(source)!=entry['candidate_sha256']:raise ValueError('Prepared parent changed')
        text=source.read_text()
        if name=='region_boundary_words.h':
            if text.count(before)!=1 or text.count(transforms[1][0])!=2:
                raise ValueError('Unexpected boundary readers')
            for old,new in transforms:text=text.replace(old,new)
        target=record/name;target.write_text(text)
        files[name]=dict(baseline_sha256=entry['baseline_sha256'],candidate_sha256=sha(target))
    result=dict(plan)
    result.update(complete=False,numerics_verified=False,files=files,
                  parent=str(args.parent.resolve()),parent_proof_sha256=sha(parent_proof),
                  preparation_script=str(Path(__file__).resolve()),preparation_script_sha256=sha(Path(__file__)),
                  changes=plan['changes']+[dict(before=old,after=new,count=1 if old==before else 2) for old,new in transforms],
                  fp32_prefetch_words=49153,
                  scope='Bounded contiguous FP32 word prefetch; same two-pass dynamic scales and exact quantization. Odd offsets use one leading half-word. Weight URAM and local output word packing retained; actual Burst/URAM/II and hardware numerics unverified.')
    (record/'preparation.json').write_text(json.dumps(result,indent=2)+'\n')
    print('U02_REGION_BOUNDARY_PREFETCH_PREPARATION',record)


if __name__=='__main__':main()
