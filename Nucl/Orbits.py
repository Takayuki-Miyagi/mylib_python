#!/usr/bin/env python3
class Orbit:
    def __init__(self):
        self.n = -1
        self.l = -1
        self.j = -1
        self.z = -1
        self.e = -1
    def set_orbit(self, *nljz):
        self.n, self.l, self.j, self.z = nljz
        self.e = 2*self.n+self.l
    def get_nljz(self):
        return (self.n, self.l, self.j, self.z)

class OrbitIsospin:
    def __init__(self):
        self.n = -1
        self.l = -1
        self.j = -1
        self.e = -1
    def set_orbit(self, *nlj):
        self.n, self.l, self.j = nlj
        self.e = 2*self.n+self.l
    def get_nlj(self):
        return (self.n, self.l, self.j)

class Orbits:
    def __init__(self, emax=None, lmax=None, shell_model_space=None, verbose=False):
        self.nljz_idx = {}
        self.orbits  = []
        self.norbs = -1
        self.emax = -1
        self.lmax = -1
        self._labels_orbital_angular_momentum = ('s',\
                'p','d','f','g','h','i','k','l','m','n',\
                'o','q','r','t','u','v','w','x','y','z')
        self.verbose=verbose
        if( emax == None and lmax==None and shell_model_space==None): return
        self.set_orbits(emax=emax,lmax=lmax,shell_model_space=shell_model_space)
    def add_orbit(self,*nljz):
        if(nljz in self.nljz_idx):
            if(self.verbose): print("The orbit ({:3d},{:3d},{:3d},{:3d}) is already there.".format(*nljz) )
            return
        self.norbs = len(self.orbits)+1
        idx = self.norbs
        self.nljz_idx[nljz] = idx
        orb = Orbit()
        orb.set_orbit(*nljz)
        self.orbits.append( orb )
        self.emax = max(self.emax, 2*nljz[0]+nljz[1])
        self.lmax = max(self.lmax, nljz[1])
    def get_orbit_label(self, idx):
        return get_orbit_label_from_orbit(self.get_orbit(idx))
    def get_orbit_label_from_orbit(self, o):
        if(o.z==-1): return f'p{o.n}{self._labels_orbital_angular_momentum[o.l]}{o.j}/2'
        if(o.z== 1): return f'n{o.n}{self._labels_orbital_angular_momentum[o.l]}{o.j}/2'
        return

    def add_orbit_from_label(self,string):
        """
        string format should be like p0s1 => proton 0s1/2
        """
        pn = string[0]
        if( pn == "p" ): z=-1
        elif( pn == "n" ): z=1
        else:
            print( "parse error in add_orbit_from_label: "+ string )
            return
        nlj_str = string[1:]
        import re
        l_str = re.findall('[a-z]+',nlj_str)[0]
        n_str, j_str = re.findall('[0-9]+',nlj_str)
        n = int(n_str)
        l = 0
        for l_label in self._labels_orbital_angular_momentum:
            if(l_str == l_label): break
            l += 1
        j = int(j_str)
        self.add_orbit(n,l,j,z)
    def add_orbits_from_labels(self,*strings):
        for label in strings:
            self.add_orbit_from_label(label)
    def get_orbit(self,idx):
        return self.orbits[idx-1]
    def get_orbit_label(self,idx):
        o = self.get_orbit(idx)
        pn = "p"
        if(o.z==1): pn="n"
        return pn+str(o.n)+self._labels_orbital_angular_momentum[o.l]+str(o.j)
    def get_orbit_index(self,*nljz):
        return self.nljz_idx[nljz]
    def get_orbit_index_from_orbit(self,o):
        return self.get_orbit_index(o.n,o.l,o.j,o.z)
    def get_orbit_index_from_tuple(self,nljz):
        return self.nljz_idx[nljz]
    def get_num_orbits(self):
        return self.norbs
    def set_orbits(self, emax=None, lmax=None, shell_model_space=None, order_pn=False):
        if(order_pn):
            if( emax != None):
                if( lmax==None ): lmax=emax
                for z in [-1,1]:
                    for N in range(emax+1):
                        for l in range(min(N+1,lmax)+1):
                            if( (N-l)%2 == 1 ): continue
                            n = (N-l)//2
                            for j in [2*l-1, 2*l+1]:
                                if( j<0 ): continue
                                self.add_orbit( n,l,j,z )
        else:
            if( emax != None):
                if( lmax==None ): lmax=emax
                for N in range(emax+1):
                    for l in range(min(N+1,lmax)+1):
                        if( (N-l)%2 == 1 ): continue
                        n = (N-l)//2
                        for j in [2*l-1, 2*l+1]:
                            if( j<0 ): continue
                            for z in [-1,1]:
                                self.add_orbit( n,l,j,z )
        if( shell_model_space != None ):
            if( shell_model_space == "p-shell" ):
                self.add_orbits_from_labels( "p0p3","p0p1","n0p3","n0p1" )
            if( shell_model_space == "sd-shell" ):
                self.add_orbits_from_labels( "p0d5","p1s1","p0d3","n0d5","n1s1","n0d3" )
            if( shell_model_space == "pf-shell" ):
                self.add_orbits_from_labels( "p0f7","p1p3","p1p1","p0f5","n0f7","n1p3","n1p1","n0f5" )
    def is_same_orbit(self, oi, oj):
        if(oi.n != oj.n): return False
        if(oi.l != oj.l): return False
        if(oi.j != oj.j): return False
        if(oi.z != oj.z): return False
        return True
    def __str__(self):
        return self.print_orbits()
    def print_orbits(self):
        string = "Orbits list:\n"
        string += "idx,  n,  l,  j,  z,  e\n"
        for o in self.orbits:
            nljz = o.get_nljz()
            n,l,j,z = o.get_nljz()
            idx = self.get_orbit_index( n,l,j,z )
            idx = self.get_orbit_index_from_tuple( nljz )
            idx = self.get_orbit_index_from_orbit( o )
            string += "{:3d},{:3d},{:3d},{:3d},{:3d},{:3d}\n".format(idx,*nljz,o.e)
        return string[:-1]

class OrbitsIsospin:
    def __init__(self, emax=None, lmax=None, shell_model_space=None, verbose=False):
        self.nlj_idx = {}
        self.orbits  = []
        self.norbs = -1
        self.emax = -1
        self.lmax = -1
        self.verbose=verbose
        self._labels_orbital_angular_momentum = ('s',\
                'p','d','f','g','h','i','k','l','m','n',\
                'o','q','r','t','u','v','w','x','y','z')
        if( emax == None and lmax==None and shell_model_space==None): return
        self.set_orbits(emax=emax,lmax=lmax,shell_model_space=shell_model_space)
    def add_orbit(self,*nlj):
        if(nlj in self.nlj_idx):
            if(self.verbose): print("The orbit ({:3d},{:3d},{:3d}) is already there.".format(*nlj) )
            return
        self.norbs = len(self.orbits)+1
        idx = self.norbs
        self.nlj_idx[nlj] = idx
        orb = OrbitIsospin()
        orb.set_orbit(*nlj)
        self.orbits.append( orb )
        self.emax = max(self.emax, 2*nlj[0]+nlj[1])
        self.lmax = max(self.lmax, nlj[1])
    def add_orbit_from_label(self,string):
        """
        string format should be like 0s1 => 0s1/2
        """
        nlj_str = string
        import re
        l_str = re.findall('[a-z]+',nlj_str)[0]
        n_str, j_str = re.findall('[0-9]+',nlj_str)
        n = int(n_str)
        l = 0
        for l_label in self._labels_orbital_angular_momentum:
            if(l_str == l_label): break
            l += 1
        j = int(j_str)
        self.add_orbit(n,l,j)
    def add_orbits_from_labels(self,*strings):
        for label in strings:
            self.add_orbit_from_label(label)
    def get_orbit(self,idx):
        return self.orbits[idx-1]
    def get_orbit_index(self,*nlj):
        return self.nlj_idx[nlj]
    def get_orbit_index_from_orbit(self,o):
        return self.get_orbit_index(o.n,o.l,o.j)
    def get_orbit_index_from_tuple(self,nlj):
        return self.nlj_idx[nlj]
    def get_num_orbits(self):
        return self.norbs
    def set_orbits(self, emax=None, lmax=None, shell_model_space=None):
        if( emax != None):
            if( lmax==None ): lmax=emax
            for N in range(emax+1):
                for l in range(min(N+1,lmax)+1):
                    if( (N-l)%2 == 1 ): continue
                    n = (N-l)//2
                    for j in [2*l-1, 2*l+1]:
                        if( j<0 ): continue
                        self.add_orbit( n,l,j )
        if( shell_model_space != None ):
            if( shell_model_space == "p-shell" ):
                self.add_orbits_from_labels( "0p3","0p1" )
            if( shell_model_space == "sd-shell" ):
                self.add_orbits_from_labels( "0d5","1s1","0d3" )
            if( shell_model_space == "pf-shell" ):
                self.add_orbits_from_labels( "0f7","1p3","1p1","0f5" )
    def print_orbits(self):
        print("Orbits (Isospin symmetry assumed) list:")
        print("idx,  n,  l,  j,  e")
        for o in self.orbits:
            nlj = o.get_nlj()
            n,l,j = o.get_nlj()
            idx = self.get_orbit_index( n,l,j )
            idx = self.get_orbit_index_from_tuple( nlj )
            idx = self.get_orbit_index_from_orbit( o )
            print("{:3d},{:3d},{:3d},{:3d},{:3d}".format(idx,*nlj,o.e) )

def main():
    orbs = Orbits()
    orbs.set_orbits(emax=0)
    orbs.print_orbits()

if(__name__ == "__main__"):
    main()
