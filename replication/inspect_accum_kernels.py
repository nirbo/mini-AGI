"""Disassemble embedded QuACK cubins using the venv's existing nvdisasm.

This is CPU-only binary inspection. It neither loads nor executes CUDA code.
The AOT .o contains a host ELF and a nested CUDA ELF; extract only the latter.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess


def cuda_elf(data):
    start=0
    while True:
        offset=data.find(b'\x7fELF',start)
        if offset<0:raise ValueError('no embedded CUDA ELF found')
        start=offset+4
        if data[offset+4:offset+6]!=b'\x02\x01':continue  # ELF64 little endian
        h=struct.unpack_from('<16sHHIQQQIHHHHHH',data,offset)
        if h[2]!=190:continue  # EM_CUDA
        end=max(h[8],h[5]+h[9]*h[10],h[6]+h[11]*h[12])
        for i in range(h[12]):
            section=struct.unpack_from('<IIQQQQIIQQ',data,offset+h[6]+i*h[11])
            if section[1]!=8:  # SHT_NOBITS has no file bytes
                end=max(end,section[4]+section[5])
        if offset+end>len(data):raise ValueError('truncated embedded CUDA ELF')
        return data[offset:offset+end]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cache',type=Path,default=Path('runs/accum-quack-cache'))
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    directory=args.out.with_suffix('');directory.mkdir(parents=True,exist_ok=True)
    disassembler=Path('.venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/nvdisasm')
    objects=[]
    for path in sorted(args.cache.rglob('*.o')):
        raw=path.read_bytes();cubin=cuda_elf(raw)
        tag=hashlib.sha256(cubin).hexdigest()
        binary=directory/(tag+'.cubin');binary.write_bytes(cubin)
        result=subprocess.run([str(disassembler),str(binary)],text=True,capture_output=True,check=True)
        sass=directory/(tag+'.sass');sass.write_text(result.stdout)
        instructions=sorted(set(re.findall(r'\b(?:H|Q)MMA\.[A-Za-z0-9_.]+',result.stdout)))
        objects.append(dict(object=str(path),object_sha256=hashlib.sha256(raw).hexdigest(),
                            cubin=str(binary),cubin_sha256=tag,sass=str(sass),
                            sass_sha256=hashlib.sha256(result.stdout.encode()).hexdigest(),mma_instructions=instructions))
    found={op for row in objects for op in row['mma_instructions']}
    if not {'HMMA.16816.F16','HMMA.16816.F32'} <= found:
        raise AssertionError(f'both accumulator instructions must be present: {found}')
    args.out.write_text(json.dumps(dict(status='passed',objects=objects,
                         disassembler=str(disassembler),disassembler_sha256=hashlib.sha256(disassembler.read_bytes()).hexdigest(),
                         instructions=sorted(found)),indent=2)+'\n')
    print(json.dumps(dict(object_count=len(objects),instructions=sorted(found))))


if __name__=='__main__':main()
