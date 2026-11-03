"""Package the proven RTL under its own name; DSP attributes only, no arithmetic edits."""
import argparse
import json
from pathlib import Path
import re
from prepare_stream_ip import sha, multiplier_leaves, verify_register_map

TOP='u02_stream_128write_v2'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    args=parser.parse_args()
    assert not args.manifest.exists()
    solution=args.project.resolve()/'zu5ev';source=solution/'syn/verilog';package=solution/'impl/ip'
    registers=verify_register_map(source,TOP,'C_V')
    leaves=multiplier_leaves(source,'requant_words',16,16,dw_taps=2,pw_input_lanes=32,
                             top=TOP,multiplier_prefix='u02_stream_128wri')
    top=(source/(TOP+'.v')).read_text()
    assert re.search(r'C_M_AXI_RESULTS_DATA_WIDTH\s*=\s*128;',top)
    records=[]
    for path in sorted(source.glob('*.v')):
        raw=path.read_bytes();text=raw.decode('ascii')
        if path.name in leaves:
            needle='module '+leaves[path.name]+'('
            assert text.count(needle)==1
            text=text.replace(needle,'(* use_dsp = "yes" *)\n'+needle)
        target=package/'hdl/verilog'/path.name
        assert target.is_file()
        target.write_bytes(text.encode('ascii'))
        records.append(dict(file=path.name,source_sha256=sha(raw),package_sha256=sha(target.read_bytes()),
                            dsp_leaf=leaves.get(path.name)))
    script=package/'run_ippack.tcl';text=script.read_text();needle='set_property core_revision $Revision $core'
    if needle in text:
        backup=package/'run_ippack.original.tcl';assert not backup.exists()
        backup.write_bytes(script.read_bytes());script.write_text(text.replace(needle,'set_property core_revision 1 $core'))
    else:assert 'set_property core_revision 1 $core' in text
    result=dict(source_modified=False,mapping_only=True,top=TOP,axi_results_bits=128,
                channel_lanes=16,pw_input_lanes=32,pw_output_lanes=16,dw_taps_per_iteration=2,
                pw_int8_multipliers=512,dw_int8_multipliers=32,requant_multipliers=16,
                requant_module='requant_words',register_map_verified=registers,files=records,
                package_tcl_sha256=sha(script.read_bytes()))
    assert all(sha((source/r['file']).read_bytes())==r['source_sha256'] for r in records)
    args.manifest.write_text(json.dumps(result,indent=2)+'\n')
    print('DW128_V2_IP_PREPARED_ORIGINAL_RTL_UNCHANGED',flush=True)


if __name__=='__main__':main()
