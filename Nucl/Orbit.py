#!/usr/bin/env python3
class Orbit:
    def __init__(self):
        self.n = -1
        self.l = -1
        self.j = -1
        self.z = -1
    def set_orbit(self,n,l,j,z):
        self.n = n
        self.l = l
        self.j = j
        self.z = z

class Orbits:
    def __init__(self):
        self.nljz_idx = {}
        self.idx_orb  = {}
        self.norbs = -1
    def add_orbit(self,n,l,j,z,idx):
        self.nljz_idx[(n,l,j,z)] = idx
        orb = Orbit()
        orb.set_orbit(n,l,j,z)
        self.idx_orb[idx] = orb
        self.norbs = max(self.norbs, idx)
    def get_orbit(self,idx):
        return self.idx_orb[idx]

def main():
    orbs = Orbits()
    orbs.add_orbit(0,0,1,-1,1)
    orbs.add_orbit(0,0,1, 1,2)
    idx = 1
    orb = orbs.get_orbit(idx)
    print(orb.n,orb.l,orb.j,orb.z)
    idx = 2
    orb = orbs.get_orbit(idx)
    print(orb.n,orb.l,orb.j,orb.z)


if(__name__ == "__main__"):
    main()
