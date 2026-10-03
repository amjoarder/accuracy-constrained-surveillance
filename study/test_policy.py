import unittest
try:
 from policy import choose, pareto
except ImportError:
 choose=pareto=None

class PolicyTests(unittest.TestCase):
 def test_each_task_and_both_metrics_are_constrained(self):
  self.assertIsNotNone(choose,'Policy implementation is not yet present')
  rows={'object':[{'id':'o0','ap50':.8,'ap95':.6,'ms':20},{'id':'bad','ap50':.79,'ap95':.4,'ms':1},{'id':'good','ap50':.77,'ap95':.58,'ms':10}],
        'behavior':[{'id':'b0','ap50':.9,'ap95':.5,'ms':20},{'id':'b1','ap50':.86,'ap95':.48,'ms':8}]}
  result=choose(rows,{'object':'o0','behavior':'b0'},.95)
  self.assertEqual(result['object']['id'],'good')
  self.assertEqual(result['behavior']['id'],'b1')
  self.assertEqual(result['estimated_ms'],18)
 def test_dominance_preserves_distinct_tradeoffs_and_ties(self):
  self.assertIsNotNone(pareto,'Pareto implementation is not yet present')
  rows=[{'id':'a','ms':10,'ap50':.8,'ap95':.6},{'id':'b','ms':12,'ap50':.7,'ap95':.5},{'id':'c','ms':8,'ap50':.75,'ap95':.55}]
  self.assertEqual({r['id'] for r in pareto(rows)},{'a','c'})
 def test_infeasible_requirements_are_explicit(self):
  self.assertIsNotNone(choose,'Policy implementation is not yet present')
  rows={'object':[{'id':'a','ap50':.8,'ap95':.6,'ms':10}],'behavior':[{'id':'b','ap50':.9,'ap95':.5,'ms':10}]}
  with self.assertRaises(ValueError):choose(rows,{'object':'a','behavior':'b'},1.01)
if __name__=='__main__':unittest.main()
