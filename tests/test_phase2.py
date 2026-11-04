"""Small regression checks, independent of production data or held-out sets."""
import unittest, tempfile, runpy, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from decoy import models, attacker

class Phase2(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.P=SimpleNamespace(nc=1,ne=1,nr=2,W=3,H=5,
          tr=np.random.default_rng(0).normal(size=(6,8,4)).astype('float32'),
          va=np.random.default_rng(1).normal(size=(6,8,4)).astype('float32'))
    def tearDown(self):self.tmp.cleanup()
    def hp(self,key):return {'B4':dict(hidden=4,layers=1),'B5':dict(hidden=4,noise=2),
        'B6':dict(hidden=8,latent=2),'B7':dict(hidden=4,layers=2),
        'B8':dict(d_model=8,layers=1),'B9':dict(hidden=4,layers=1),'C1':dict(hidden=4,layers=1),'A3':dict(hidden=4,layers=1)}[key]
    def test_gan_exact_resume_all_phases(self):
        for key in ('B5','B6','B7'):
            def build():torch.manual_seed(42);return models.build(key,self.P,self.hp(key))
            straight=build();models.train_gan(straight,self.P.tr,.001,4,2,42,self.root/(key+'full.pt'),'cpu',every=1)
            split=build();path=self.root/(key+'resume.pt')
            for stop in range(1,split.n_phases*2+1):
                split=build();models.train_gan(split,self.P.tr,.001,4,2,42,path,'cpu',every=1,stop_after=stop)
            for name,a in straight.state_dict().items():
                self.assertTrue(torch.equal(a,split.state_dict()[name]),(key,name))
    def test_no_future_response_leakage(self):
        for key in ('B4','B5','B6','B7','B8','B9','C1','A3'):
            m=models.build(key,self.P,self.hp(key));x=self.P.va.copy();x[:,3:,2:]+=1000
            a=models.generate(m,self.P.va,'cpu',7);b=models.generate(m,x,'cpu',7)
            np.testing.assert_array_equal(a,b)
            np.testing.assert_array_equal(a[...,1],self.P.va[:,3:,1])
            self.assertEqual(a.shape,(6,5,4))
    def test_transformer_causality(self):
        m=models.build('B8',self.P,self.hp('B8'));x=self.P.va.copy();x[:,6:,:2]+=1000
        a=models.generate(m,self.P.va,'cpu');b=models.generate(m,x,'cpu')
        np.testing.assert_allclose(a[:,:3,2:],b[:,:3,2:],atol=1e-6)
    def test_purge(self):
        p=SimpleNamespace(W=3,H=5,info={'window_index':{'val':[('run',i*4) for i in range(20)]}})
        left,right=attacker.partition(p)
        self.assertTrue(left[-1]*4+8<=right[0]*4)
    def test_invalid_validation_no_checkpoint(self):
        m=models.build('C1',self.P,self.hp('C1'));va=self.P.va.copy();va[:,3:,2:]=np.nan
        with self.assertRaises(FloatingPointError):models.train(m,self.P.tr,va,.001,4,2,1,5,0,self.root/'bad.pt','cpu')
        self.assertFalse((self.root/'bad.pt').exists())
    def test_cache_identity_and_selftest(self):
        run=runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/02_train.py'))
        for _ in range(2):run['selftest'](self.P,{'batch':4},self.root,torch.device('cpu'))
        run['guard_run'](self.root/'cache','C1',self.P,{'hidden':4},{},0)
        run['guard_run'](self.root/'cache','C1',self.P,{'hidden':4},{},0)
        with self.assertRaises(RuntimeError):run['guard_run'](self.root/'cache','C1',self.P,{'hidden':8},{},0)

if __name__=='__main__':unittest.main()
