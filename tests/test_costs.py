"""Check the documented rates reproduce all 24 published rounded costs."""
from pathlib import Path
import unittest
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
class Costs(unittest.TestCase):
 def test_published_costs(self):
  expected={'DeepSeek V4 Pro':[.0357,.0180,.0321,.0151],'Qwen3-VL-235B-A22B-Instruct':[.0428,.0124,.0337,.0092],'Kimi K2.6':[.1196,.0431,.0948,.0297],'Grok 4.3':[.1608,.0400,.1432,.0268],'GPT-5.2':[.3353,.0759,.2971,.0563],'Claude Opus 4.8':[1.5909,.3080,1.4546,.2249]}
  d=pd.read_csv(ROOT/'results/tables/efficiency_by_condition.csv').set_index(['model','exam','strategy'])
  for m,vals in expected.items():
   for (e,s),v in zip([('FIR','item-wise'),('FIR','whole-exam'),('BIR','item-wise'),('BIR','whole-exam')],vals):
    with self.subTest(model=m,exam=e,strategy=s):self.assertEqual(round(float(d.loc[(m,e,s),'estimated_api_cost_usd']),4),v)
