import math
import random
from types import SimpleNamespace
import pytest
from nanobody_opt import pr100_psh_geometry as geometry

class Residue:
    def __init__(self, coordinates):
        self.atoms=[SimpleNamespace(coords=xyz) for xyz in coordinates]
    def distance(self,left,right):
        return original_reference(left,right)
    def sb_distance(self,*args):
        return 'unchanged salt bridge'
class Chain:
    def __init__(self,residues):
        self.residues=dict(enumerate(residues))
def modules():
    return SimpleNamespace(),SimpleNamespace(Residue=Residue,PDBchain=Chain)
def original_reference(a,b):
    result=99999999
    for left in a.atoms:
        for right in b.atoms:
            x,y,z=(left.coords[i]-right.coords[i] for i in range(3))
            v=math.sqrt(x*x+y*y+z*z)
            if v<result:result=v
    return result

def test_cache_reuses_directions_and_preserves_results():
    a,b=Residue([(0.,0.,0.)]),Residue([(3.,4.,0.)])
    original,sb=Residue.distance,Residue.sb_distance
    surface,module=modules()
    with geometry.accelerated_distances(surface,module,require_pinned=False) as stats:
        assert a.distance(a,b)==b.distance(b,a)==a.distance(a,b)==5
    assert stats['reference_calls']==1 and stats['cache_hits']==2
    assert Residue.distance is original and Residue.sb_distance is sb

def test_numpy_matches_original_bitwise_on_random_residues():
    rng=random.Random(21)
    residues=[Residue([tuple(rng.uniform(-50,50)for _ in range(3))for _ in range(rng.randrange(1,18))])for _ in range(50)]
    expected=[[r.distance(r,s)for s in residues]for r in residues]
    table=geometry._distance_table(residues,'numpy',2048)
    assert [[float(v).hex()for v in row]for row in table]==[[float(v).hex()for v in row]for row in expected]

def test_all_atoms_not_representative_ca():
    a,b=Residue([(0.,0.,0.),(10.,0.,0.)]),Residue([(20.,0.,0.),(10.5,0.,0.)])
    assert geometry._distance_table([a,b],'numpy',100)[0][1]==0.5

@pytest.mark.parametrize('distance',[math.nextafter(4.,0.),4.,math.nextafter(4.,math.inf),math.nextafter(7.5,0.),7.5,math.nextafter(7.5,math.inf),0.,99999999.,100000000.])
def test_cutoff_and_sentinel_semantics(distance):
    a,b=Residue([(0.,0.,0.)]),Residue([(distance,0.,0.)])
    expected=a.distance(a,b);actual=geometry._distance_table([a,b],'numpy',100)[0][1]
    assert float(actual).hex()==float(expected).hex()
    assert (actual<4)==(expected<4)and(actual<=7.5)==(expected<=7.5)

def test_chain_integration_and_exception_restore():
    a,b=Residue([(0.,0.,0.)]),Residue([(3.,4.,0.)])
    surface,module=modules();original,init=Residue.distance,Chain.__init__
    with pytest.raises(KeyboardInterrupt):
        with geometry.accelerated_distances(surface,module,backend='numpy',require_pinned=False)as stats:
            Chain([a,b]);assert b.distance(b,a)==5
            assert stats['tables_built']==1 and stats['table_hits']==1
            raise KeyboardInterrupt()
    assert Residue.distance is original and Chain.__init__ is init

@pytest.mark.parametrize('residues,limit',[([Residue([])],10),([Residue([(float('nan'),0.,0.)])],10),([Residue([(1e308,0.,0.)])],10),([Residue([(0.,0.,0.),(1.,0.,0.)])],1)])
def test_unsupported_and_oversized_fall_back(residues,limit):
    surface,module=modules()
    with geometry.accelerated_distances(surface,module,backend='numpy',max_atoms=limit,require_pinned=False)as stats:
        Chain(residues)
        assert residues[0].distance(residues[0],residues[0])==original_reference(residues[0],residues[0])
    assert stats['tables_built']==0 and stats['table_fallbacks']==1

def test_nested_installation_rejected_and_outer_restored():
    surface,module=modules();original=Residue.distance
    with geometry.accelerated_distances(surface,module,require_pinned=False):
        with pytest.raises(RuntimeError,match='already_installed'):
            with geometry.accelerated_distances(surface,module,require_pinned=False):pass
    assert Residue.distance is original


def test_unavailable_cuda_raises_without_leaving_patch(monkeypatch):
    import sys
    fake=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:False))
    monkeypatch.setitem(sys.modules,'torch',fake)
    surface,module=modules();original,init=Residue.distance,Chain.__init__
    with pytest.raises(RuntimeError,match='cuda_not_available'):
        with geometry.accelerated_distances(surface,module,backend='cuda',require_pinned=False):
            Chain([Residue([(0.,0.,0.)])])
    assert Residue.distance is original and Chain.__init__ is init


@pytest.mark.parametrize('backend,max_atoms',[('approximate_ca',100),('numpy',0),('numpy',True)])
def test_invalid_options_leave_originals_untouched(backend,max_atoms):
    surface,module=modules();original=Residue.distance
    with pytest.raises(ValueError):
        with geometry.accelerated_distances(surface,module,backend=backend,max_atoms=max_atoms,require_pinned=False):pass
    assert Residue.distance is original


def test_incomplete_module_does_not_leak_context_lock():
    surface,module=modules()
    with pytest.raises(AttributeError):
        with geometry.accelerated_distances(surface,SimpleNamespace(),require_pinned=False):pass
    with geometry.accelerated_distances(surface,module,require_pinned=False):pass
