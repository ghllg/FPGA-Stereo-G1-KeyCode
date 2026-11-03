"""Package the single-PW 16x32 candidate without changing its frozen HLS RTL.

Uses the existing register, hierarchy and signed multiplier verifiers. The
older packager and its default multiplier expectations remain unchanged.
"""
import argparse
import json
from pathlib import Path

from prepare_stream_ip import sha, multiplier_leaves, verify_register_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--requant-module',choices=('requant_words',),required=True)
    parser.add_argument('--channel-lanes',type=int,choices=(16,),required=True)
    parser.add_argument('--requant-lanes',type=int,choices=(16,),required=True)
    args = parser.parse_args()
    solution=args.project.resolve()/'zu5ev'
    source,package=solution/'syn/verilog',solution/'impl/ip'
    registers=verify_register_map(source)
    leaves=multiplier_leaves(source,'requant_words',16,16,dw_taps=2,pw_input_lanes=32)
    records=[]
    for path in sorted(source.glob('*.v')):
        raw=path.read_bytes()
        text=raw.decode('ascii')
        if path.name in leaves:
            needle='module '+leaves[path.name]+'('
            assert text.count(needle)==1
            text=text.replace(needle,'(* use_dsp = "yes" *)\n'+needle)
        target=package/'hdl/verilog'/path.name
        assert target.is_file()
        target.write_bytes(text.encode('ascii'))
        records.append(dict(file=path.name,source_sha256=sha(raw),
                            package_sha256=sha(target.read_bytes()),dsp_leaf=leaves.get(path.name)))
    script=package/'run_ippack.tcl'
    text=script.read_text()
    needle='set_property core_revision $Revision $core'
    if needle in text:
        backup=package/'run_ippack.original.tcl'
        assert not backup.exists(), 'Preserve a prior packaging attempt'
        backup.write_bytes(script.read_bytes())
        script.write_text(text.replace(needle,'set_property core_revision 1 $core'))
    else:
        assert 'set_property core_revision 1 $core' in text
    result=dict(source_modified=False,mapping_only=True,channel_lanes=16,
                dw_taps_per_iteration=2,pw_input_lanes=32,pw_output_lanes=16,pw_int8_multipliers=512,dw_int8_multipliers=32,
                requant_multipliers=16,requant_module='requant_words',
                register_map_verified=registers,files=records,package_tcl_sha256=sha(script.read_bytes()))
    assert all(sha((source/item['file']).read_bytes())==item['source_sha256'] for item in records)
    args.manifest.write_text(json.dumps(result,indent=2)+'\n')
    print('U02_PW32_IP_PREPARED ORIGINAL_RTL_UNCHANGED',flush=True)


if __name__=='__main__':
    main()
