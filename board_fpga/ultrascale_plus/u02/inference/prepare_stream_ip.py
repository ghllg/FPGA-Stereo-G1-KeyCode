"""Package-only revision fix and DSP mapping attributes; preserve HLS RTL."""
import argparse,hashlib,json,re
from pathlib import Path
U02=Path(__file__).resolve().parents[1]
def sha(data):return hashlib.sha256(data).hexdigest()
def verify_compute_sharing(source,requant_module='requant_tile',top='u02_stream'):
    counts={}
    def visit(module,ancestors):
        assert module not in ancestors,('Recursive RTL hierarchy',module)
        counts[module]=counts.get(module,0)+1
        path=source/(module+'.v')
        if not path.exists():return
        # HLS function instances are not parameterized. Arithmetic primitive
        # instances use #(...), and are counted separately by multiplier_leaves.
        for child,instance in re.findall(r'(?m)^\s*(\w+)\s+(\w+)\s*\(',path.read_text()):
            if child!='module' and (source/(child+'.v')).exists():
                visit(child,ancestors+(module,))
    visit(top,())
    for name in ('pw_mac','dw_mac',requant_module):
        actual=sum(count for module,count in counts.items() if re.fullmatch(name+r'(?:_\d+)?',module))
        assert actual==1,('Compute hierarchy instance count',name,actual)
    return counts
def verify_register_map(source,top='u02_stream',c_symbol='C'):
    control=(source/(top+'_control_s_axi.v')).read_text()
    api=(U02/'board/linux/u02_stream_api.h').read_text()
    register_map={}
    for name in ('RETURN','A','B','C','M','N','K'):
        rtl_name='ADDR_AP_RETURN_0' if name=='RETURN' else 'ADDR_'+(c_symbol if name=='C' else name)+'_DATA_0'
        actual=int(re.search(r'\b'+rtl_name+r"\s*=\s*7'h([0-9a-fA-F]+)",control).group(1),16)
        expected=int(re.search(r'#define U02_STREAM_REG_'+name+r'\s+(0x[0-9a-f]+)',api).group(1),16)
        assert actual==expected,('Driver/RTL register mismatch',name,actual,expected)
        register_map[name]=actual
    return register_map
def multiplier_leaves(source,requant_module='requant_tile',channel_lanes=8,requant_lanes=4,dw_taps=1,pw_input_lanes=16,
                      top='u02_stream',multiplier_prefix='u02_stream_mul_'):
    assert channel_lanes in (8,16), 'Unsupported verified channel parallelism'
    assert requant_lanes in (4,8,16), 'Unsupported verified requant parallelism'
    assert dw_taps in (1,2), 'Unsupported DW tap parallelism'
    assert pw_input_lanes in (16,32), 'Unsupported PW input parallelism'
    # HLS-generated suffixes change when scheduling changes. Identify the
    # arithmetic by the actual signed leaf widths and verify instance counts.
    leaves={}
    verify_compute_sharing(source,requant_module,top)
    # Reject an accidental second PW/DW implementation before packaging.
    for name in ('pw_mac','dw_mac'):
        modules=sorted(p.name for p in source.glob(name+'*.v')
                       if re.fullmatch(name+r'(?:_\d+)?\.v',p.name))
        assert modules==[name+'.v'],('Duplicated MAC modules',name,modules)
    for filename,widths,expected in [('pw_mac.v',(8,8,16),channel_lanes*pw_input_lanes),('dw_mac.v',(8,8,16),channel_lanes*dw_taps),
                                     (requant_module+'.v',(32,24,56),requant_lanes)]:
        instances=re.findall(r'\b('+re.escape(multiplier_prefix)+r'\w+)\s*#', (source/filename).read_text())
        matched=0
        for module in set(instances):
            raw=(source/(module+'.v')).read_text()
            ports=[('input',widths[0],'a'),('input',widths[1],'b'),('output',widths[2],'p')]
            if not all(re.search(r'\b'+direction+r'\s*\[\s*'+str(width)+r'\s*-\s*1\s*:\s*0\s*\]\s*'+port+r'\s*;',raw)
                       for direction,width,port in ports):continue
            leaf=re.findall(r'module\s+(\w+_MulnS_\d+)\(clk, ce, a, b, p\);',raw)
            assert len(leaf)==1 and 'assign tmp_product = a_reg0 * b_reg0;' in raw
            assert re.search(r'reg signed\s*\[\s*'+str(widths[0])+r'\s*-\s*1\s*:\s*0\s*\]\s*a_reg0;',raw)
            assert re.search(r'reg signed\s*\[\s*'+str(widths[1])+r'\s*-\s*1\s*:\s*0\s*\]\s*b_reg0;',raw)
            leaves[module+'.v']=leaf[0];matched+=instances.count(module)
        assert matched==expected,(filename,matched,expected)
    return leaves

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--project',type=Path,default=U02/'inference/stream_hls_03')
    parser.add_argument('--manifest',type=Path,default=U02/'reports/stream_ip_preparation.json')
    parser.add_argument('--requant-module',choices=('requant_tile','requant_words'),default='requant_tile')
    parser.add_argument('--channel-lanes',type=int,choices=(8,16),default=8)
    parser.add_argument('--requant-lanes',type=int,choices=(4,8,16),default=4)
    args=parser.parse_args()
    solution=args.project.resolve()/'zu5ev';source=solution/'syn/verilog';package=solution/'impl/ip'
    register_map=verify_register_map(source)
    leaves=multiplier_leaves(source,args.requant_module,args.channel_lanes,args.requant_lanes)
    records=[]
    for p in source.glob('*.v'):
        raw=p.read_bytes();text=raw.decode('ascii')
        if p.name in leaves:
            needle='module '+leaves[p.name]+'('
            assert text.count(needle)==1 and 'assign tmp_product = a_reg0 * b_reg0;' in text
            text=text.replace(needle,'(* use_dsp = "yes" *)\n'+needle)
        target=package/'hdl/verilog'/p.name
        assert target.exists(),target
        target.write_bytes(text.encode('ascii'))
        records.append({'file':p.name,'source_sha256':sha(raw),'package_sha256':sha(target.read_bytes()),'dsp_leaf':leaves.get(p.name)})
    script=package/'run_ippack.tcl';text=script.read_text()
    needle='set_property core_revision $Revision $core'
    if needle in text:
        backup=package/'run_ippack.original.tcl'
        if not backup.exists():backup.write_bytes(script.read_bytes())
        text=text.replace(needle,'set_property core_revision 1 $core');script.write_text(text)
    else:assert 'set_property core_revision 1 $core' in text
    manifest={'source_modified':False,'mapping_only':True,'channel_lanes':args.channel_lanes,'pw_int8_multipliers':args.channel_lanes*16,'dw_int8_multipliers':args.channel_lanes,
              'requant_multipliers':args.requant_lanes,'requant_module':args.requant_module,'register_map_verified':register_map,'files':records,'package_tcl_sha256':sha(script.read_bytes())}
    args.manifest.write_text(json.dumps(manifest,indent=2))
    print('U02_STREAM_IP_PREPARED DSP leaves tagged; arithmetic/registers unchanged')
if __name__=='__main__':main()
