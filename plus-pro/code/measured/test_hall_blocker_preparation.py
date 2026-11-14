"""Small CPU correctness checks for blocker conversion and pairwise isolation."""
import unittest
import numpy as np
from prepare_hall_blocker_data import calibrate,groups_info,nearest,select_positions,split_positions,ROTATION
from verify_hall_blocker_preparation import reference_selection,verify_splits

class PreparationChecks(unittest.TestCase):
    def test_calibration_sign_shift_and_every_physical_port(self):
        csi=np.zeros((64,1024,2),np.float32)
        csi[...,0]=np.arange(64)[:,None]+1
        csi[...,1]=np.arange(1024)[None,:]/1024
        assignments=np.arange(64).reshape(2,4,8)[:,::-1,::-1]
        offsets=dict(cpo=np.linspace(.1,.7,64),sto=np.linspace(-.4,.6,64))
        prepared,_=calibrate(csi,offsets,assignments)
        self.assertEqual(prepared.shape,(2,8,4,1024))
        for panel in range(2):
            for horizontal in range(8):
                for vertical in range(4):
                    a=assignments[panel,vertical,horizontal]
                    for k in [0,1,511,512,1023]:
                        j=(k+512)%1024
                        z=complex(float(csi[a,j,0]),float(csi[a,j,1]))
                        phi=offsets["cpo"][a]+2*np.pi*offsets["sto"][a]*k/1024
                        expected=z*complex(np.cos(phi),np.sin(phi))
                        self.assertLess(abs(prepared[panel,horizontal,vertical,k]-expected),1e-5)
        wrong,_=calibrate(csi,dict(cpo=-offsets["cpo"],sto=-offsets["sto"]),assignments)
        self.assertGreater(float(np.max(np.abs(prepared-wrong))),.1)

    def test_coordinate_axes_are_a_rotation_not_a_reflection(self):
        np.testing.assert_array_equal(ROTATION@ROTATION.T,np.eye(3))
        self.assertEqual(np.linalg.det(ROTATION),1)
        np.testing.assert_array_equal(np.array([[2.,3.,7.]])@ROTATION.T,[[2.,-7.,3.]])
        np.testing.assert_array_equal(np.array([[0.,1.,0.]])@ROTATION.T,[[0.,0.,1.]])

    def test_representative_depends_only_on_position_and_index(self):
        p=np.array([[.25,.5,1],[.75,.5,1],[1.25,.5,1],[1.75,.5,1]],float)
        np.testing.assert_array_equal(select_positions(p,np.ones(4,bool),voxel=1),[0,2])
        np.testing.assert_array_equal(select_positions(p,np.array([False,True,True,True]),voxel=1),[1,2])

    def test_three_way_guard_removes_validation_near_test(self):
        p=np.array([[4*x+dx,4*y+dy,1.] for x in range(10) for y in range(10)
            for dx,dy in [(.05,.05),(3.95,.05),(.05,3.95),(3.95,3.95),(2.,2.)]])
        split=split_positions(p)
        test_count=np.zeros(len(p),int)
        self.assertTrue(any(x["validation_removed_near_test"]>0 for x in split["splits"]))
        for fold in split["splits"]:
            for a,b in [("fit","validation"),("fit","test"),("validation","test")]:
                self.assertGreaterEqual(nearest(p[fold[a]],p[fold[b]]).min(),.30)
                self.assertFalse(set(fold[a])&set(fold[b]))
            test_count[fold["test"]]+=1
        np.testing.assert_array_equal(test_count,np.ones(len(p)))

    def test_exact_duplicate_groups_do_not_conflate_unique_hashes(self):
        result=groups_info(np.array([b"a",b"b",b"a",b"c"],dtype="S64"))
        self.assertEqual(result["groups"],[[0,2]])
        self.assertEqual(result["excess_rows"],1)

    def test_independent_verifier_rejects_changed_partition(self):
        import copy
        p=np.array([[4*x+dx,4*y+dy,1.] for x in range(10) for y in range(10)
            for dx,dy in [(.05,.05),(3.95,.05),(.05,3.95),(3.95,3.95),(2.,2.)]])
        hashes=np.array([str(i).encode() for i in range(len(p))],dtype="S64")
        good=split_positions(p)
        self.assertTrue(verify_splits(p,good,hashes)["all_pairwise_guards_passed"])
        bad=copy.deepcopy(good)
        bad["splits"][0]["validation"].append(bad["splits"][0]["test"][0])
        with self.assertRaises(ValueError):verify_splits(p,bad,hashes)

    def test_independent_representative_selection_matches_ties(self):
        p=np.array([[.025,.1,1],[.175,.1,1],[.1,.1,1],[.225,.1,1]])
        valid=np.ones(4,bool)
        np.testing.assert_array_equal(reference_selection(p,valid),select_positions(p,valid))

    def test_whole_blocks_are_not_split_and_all_folds_have_support(self):
        p=np.array([[4*x+.1+.01*j,4*y+.1,1.] for x,y,n in
            [(0,-3,44),(0,-2,112),(0,-1,88),(0,0,112),(1,-5,51),(1,-4,92),(1,-3,60),(1,-2,3),(1,0,65)]
            for j in range(n)])
        split=split_positions(p);color=np.array(split["fold"]);tiles=np.floor(p[:,:2]/4).astype(int)
        for tile in np.unique(tiles,axis=0):
            self.assertEqual(len(np.unique(color[np.all(tiles==tile,axis=1)])),1)
        self.assertTrue(all(min(len(f[x]) for x in ["fit","validation","test"])>=16 for f in split["splits"]))

if __name__=="__main__":unittest.main(verbosity=2)

