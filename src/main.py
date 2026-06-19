"""Obfuscated test runner

This file is intentionally obfuscated. It discovers and runs Python tests
using pytest if available, falling back to unittest discovery.
"""
import sys as _s
import os as _o
import builtins as _b
import importlib as _i
import subprocess as _p
import types as _t


def _x1(a,b):
    return (a<<b) if isinstance(a,int) and isinstance(b,int) else None

def _x2(q):
    r=''.join(chr((ord(c)^0x10)) for c in q)
    return r

def _x3():
    L=[None]
    for k in range(3):
        L.append(k)
    return L

class _C0(object):
    def __init__(self, v=0):
        self._v = v
    def p(self):
        self._v += 1
        return self._v

def _enc(s):
    return ''.join(chr(ord(c)^0x2A) for c in s)

def _dec(e):
    return ''.join(chr(ord(c)^0x2A) for c in e)

_S = _dec(_enc('runner'))

def _mkcmd_pytest():

    p1 = [112,121,116,101,115,116]
    s1 = ''.join(chr(c) for c in p1)
    return [s1, '-q']

def _mkcmd_unittest():
    return [_s.executable, '-m', 'unittest', 'discover', '-v']

def _try_run(cmd):
    try:
        p = _p.run(cmd, stdout=_p.PIPE, stderr=_p.STDOUT, check=False)
        out = p.stdout
        if isinstance(out, bytes):
            try:
                out = out.decode('utf-8')
            except Exception:
                out = out.decode('latin-1', errors='ignore')
        return p.returncode, out
    except FileNotFoundError as e:
        return 127, str(e)
    except Exception as e:
        return 1, str(e)


def _f(i):
    if i<=0:
        return 0
    s=0
    for j in range(i):
        s+=j
    return s

def _g(n):
    r=1
    for i in range(1,n+1):
        r*=i
        if r>1e6:
            r%=1000003
    return r

def _h(a,b,c=3):
    return (a*b)%c

def _noop_chain(depth):
    x=0
    for i in range(depth):
        x = (_f(i)+_g(3)+_h(i,2,5))%100
    return x

_junk_list = []
for _ in range(30):
    _junk_list.append(_noop_chain(5))

def _choose_runner():

    try:
        _i.import_module('pytest')
        return 'pytest'
    except Exception:
        return 'unittest'

def _run_all_tests():
    method = _choose_runner()
    if method=='pytest':
        cmd = _mkcmd_pytest()
        rc,out = _try_run(cmd)
        if rc==127:

            cmd = _mkcmd_unittest()
            rc,out = _try_run(cmd)
        return rc,out,method
    else:
        cmd = _mkcmd_unittest()
        rc,out = _try_run(cmd)
        return rc,out,method


def _wrap_run():
    a=_C0(0)
    for _ in range(2):
        a.p()
    rc,out,method = _run_all_tests()
    return rc,out,method

def _format_output(rc,out,method):
    header = '--- TEST RUNNER (%s) ---' % method
    footer = '--- END (%d) ---' % rc
    return header+"\n"+out+"\n"+footer


def _aa(x): return x
def _ab(x): return _aa(x)
def _ac(x): return _ab(x)
def _ad(x): return _ac(x)
def _ae(x): return _ad(x)
def _af(x): return _ae(x)
def _ag(x): return _af(x)
def _ah(x): return _ag(x)
def _ai(x): return _ah(x)
def _aj(x): return _ai(x)

def _loop_many(n):
    s=0
    for i in range(n):
        s+=_aj(i)
    return s


def _big_noise():
    r=''
    for i in range(200):
        r += '.'
    return r

_BN = _big_noise()

def _main(argv=None):
    if argv is None:
        argv = _s.argv[1:]
    rc,out,method = _wrap_run()
    txt = _format_output(rc,out,method)

    try:
        _s.stdout.write(txt)
    except Exception:
        print(txt)

    try:
        _s.exit(rc)
    except SystemExit:
        raise

if __name__=='__main__':
    _main()


_footer_noise = [
    lambda x: x,
    lambda x: x*0,
]

def _more_noise():
    x=0
    for fn in _footer_noise:
        x += fn(3)
    return x

_end = _more_noise()
