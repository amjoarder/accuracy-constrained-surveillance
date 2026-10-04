"""Verify that the historical training records match the retained checkpoint metrics."""
from pathlib import Path
import csv,hashlib,json,unittest
BASE=Path(__file__).resolve().parent/'training'
class TrainingRecordTests(unittest.TestCase):
 def test_training_logs_match_selected_checkpoints(self):
  records=json.loads((BASE/'training_summary.json').read_text())
  for task,expected_epochs,expected_best in [('object',669,469),('behavior',489,189)]:
   with self.subTest(task=task):
    record=records[task];path=BASE/(task+'_training_results.csv')
    self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),record['results_csv_sha256'])
    self.assertEqual(hashlib.sha256((BASE/(task+'_args.yaml')).read_bytes()).hexdigest(),record['args_sha256'])
    with path.open(encoding='utf-8') as stream:rows=[{k.strip():float(v) for k,v in row.items()} for row in csv.DictReader(stream)]
    self.assertEqual(len(rows),expected_epochs)
    best=max(rows,key=lambda r:.1*r['metrics/mAP50(B)']+.9*r['metrics/mAP50-95(B)'])
    self.assertEqual(int(best['epoch']),expected_best)
    for key in ['metrics/mAP50(B)','metrics/mAP50-95(B)']:
     self.assertAlmostEqual(best[key],record['original_validation_metrics'][key],places=4)
    self.assertTrue(record['training_performed_before_deployment_study'])
    self.assertFalse(record['new_training_during_revision'])
if __name__=='__main__':unittest.main()
