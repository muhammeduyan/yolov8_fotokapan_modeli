"""Safety-routing regressions for the added adapter, using original frozen rules."""
import unittest

from ..chain import FrozenDecisionChain
from ..runtime import pixel_box


def goat(score=.99):
    return {'status':'IN_CATALOG','catalog_class':'wild_goat','agreement':True,'score':score,
            'raw_label':'id;mammalia;artiodactyla;bovidae;capra;aegagrus;wild goat'}


class RoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.chain=FrozenDecisionChain()

    def base(self,b='REVIEW',knownness=.8):
        return {'crop_id':'unseen-photo#0','candidate':'wild_goat','B_decision':b,
                'knownness':knownness,'safety':.5,'knn_distance':.3,'knn_margin':.1}

    def run_case(self,base,original,context):
        calls=[]
        def service(pixels,candidate,variant):
            calls.append((variant,pixels))
            return original if variant=='original' else context
        result=self.chain.finish(base,[10,20,50,60],[100,100],service)
        return result,calls

    def test_veto_must_not_be_rescued_by_context(self):
        result,calls=self.run_case(self.base('wild_goat'),goat(.89),goat(1.))
        self.assertEqual(result['decision'],'REVIEW')
        self.assertEqual([c[0] for c in calls],['original'])

    def test_e4_unknown_must_not_receive_context_or_become_known(self):
        outside={'status':'OUTSIDE_CATALOG','catalog_class':None,'agreement':False,'score':.99,
                 'raw_label':'id;mammalia;carnivora;canidae;canis;latrans;coyote'}
        result,calls=self.run_case(self.base(),outside,goat(1.))
        self.assertEqual(result['decision'],'UNKNOWN')
        self.assertEqual([c[0] for c in calls],['original'])

    def test_only_pending_original_review_receives_fixed_context(self):
        result,calls=self.run_case(self.base(),goat(.89),goat(.90))
        self.assertEqual(result['species'],'wild_goat')
        self.assertEqual(calls,[('original',[10,20,50,60]),('expanded',[5,15,55,65])])

    def test_blank_is_not_unknown_evidence(self):
        blank={'status':'TAXON_GROUP','catalog_class':None,'agreement':False,'score':.999,
               'raw_label':'id;;;;;;blank'}
        result,calls=self.run_case(self.base(),blank,blank)
        self.assertEqual(result['decision'],'REVIEW')
        self.assertEqual(len(calls),2)

    def test_gate_unknown_never_calls_speciesnet(self):
        result,calls=self.run_case(self.base('UNKNOWN'),goat(),goat())
        self.assertEqual(result['decision'],'UNKNOWN')
        self.assertEqual(calls,[])

    def test_edge_crop_keeps_center_without_padding(self):
        self.assertEqual(self.chain.e5b.geometry([0,0,20,30],[100,100]),[0,0,20,30])
        self.assertEqual(pixel_box([-.2,1.3,19.01,30.9],[100,100]),[0,1,20,31])


if __name__=='__main__':unittest.main()
