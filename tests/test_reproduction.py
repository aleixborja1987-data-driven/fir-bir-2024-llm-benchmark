"""Offline data and scoring tests. Run: python -m unittest discover -s tests -v"""
import unittest
from pathlib import Path
import pandas as pd
from analysis.reproduce import norm_answer,selected_answer,answer_object,read_inputs
ROOT=Path(__file__).resolve().parents[1]
class Tests(unittest.TestCase):
    def test_normalization(self):
        self.assertEqual(norm_answer('2.0'),'2');self.assertEqual(norm_answer(2),'2')
        self.assertIsNone(norm_answer('2,3'));self.assertEqual(norm_answer('blank'),'blank')
    def test_multi_option_scored_as_blank(self):
        self.assertEqual(selected_answer('2,3',{},'json'),('blank','multiple_options_to_blank'))
    def test_auxiliary_truncation(self):
        obj,text,mode=answer_object({'raw_output':'{"answer":"3","rationale":"unfinished'})
        self.assertEqual(obj['answer'],'3');self.assertEqual(mode,'unique_leading_answer_recovery')
    def test_refuse_unsupported_extraction(self):
        with self.assertRaises(ValueError):answer_object({'raw_output':'I think it is probably option 2, maybe option 3'})
    def test_inputs(self):
        k=read_inputs(ROOT);self.assertEqual(len(k),420)
        self.assertTrue(k.groupby('exam').included_in_official_200.sum().eq(200).all())
    def test_size_and_uniqueness(self):
        d=pd.read_csv(ROOT/'data/official_200_scored_items.csv')
        self.assertEqual(len(d),9600);self.assertFalse(d.duplicated(['condition_id','replicate','question_number']).any())
    def test_recovery_counts(self):
        d=pd.read_csv(ROOT/'data/reclassification_audit_scored.csv')
        self.assertEqual(len(d),54);self.assertEqual((d.recode_rule=='unique_answer_recovery').sum(),43)
        self.assertEqual(d[d.recode_rule=='unique_answer_recovery'].is_correct.sum(),37)
    def test_images(self):
        d=pd.read_csv(ROOT/'results/tables/image_hash_verification.csv');self.assertEqual(len(d),200);self.assertEqual(d.condition_id.nunique(),10);logged=d[d.logged_sha256.notna()];self.assertEqual(len(logged),180);self.assertTrue(logged.matches.eq(True).all());self.assertEqual(d.logged_sha256.isna().sum(),20)
    def test_text_stratum(self):
        d=pd.read_csv(ROOT/'results/tables/bootstrap_fir_text_only_180.csv')
        self.assertEqual(len(d),12);self.assertTrue(d.n_items.eq(180).all())
    def test_deepseek_decomposition(self):
        d=pd.read_csv(ROOT/'results/tables/bootstrap_fir_text_only_180.csv')
        g=d[d.model=='DeepSeek V4 Pro'].set_index('strategy')
        self.assertEqual(g.loc['item-wise','mean_correct'],174.5)
        self.assertEqual(g.loc['whole-exam','mean_correct'],175)
if __name__=='__main__':unittest.main()
