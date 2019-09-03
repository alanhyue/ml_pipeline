import unittest

import os
import pandas as pd
from pandas.util.testing import assert_frame_equal

from ml_pipeline import (
    Select, Impute, notation, Pipeline, PipelineUnion,
    MakeDummy,
    KeepOthers,
    Scale,
    Winsorize,
    Drop,
)

DIR = os.path.dirname(__file__)

def diff(l1, l2): return list(set(l1) - set(l2))

class TestPipelines(unittest.TestCase):

    def setUp(self):
        self.df = pd.read_csv(DIR+'/pets.csv')

    def tearDown(self):
        self.df = None
        del self.df

    def test_select_one_column(self):
        # using str
        sel = Select('pet')
        r = sel.fit_transform(self.df)
        self.assertTrue(isinstance(r, pd.DataFrame))
        self.assertTrue(r.shape[1] == 1)
        self.assertTrue(r.columns[0] == 'pet')
        # using list
        col = 'height(cm)'
        sel = Select([col])
        r = sel.fit_transform(self.df)
        self.assertTrue(isinstance(r, pd.DataFrame))
        self.assertTrue(r.shape[1] == 1)
        self.assertTrue(r.columns[0] == col)
        
    def test_select_multiple_columns(self):
        cols = ['height(cm)', 'ID', 'age']
        sel = Select(cols)
        r = sel.fit_transform(self.df)
        self.assertTrue(isinstance(r, pd.DataFrame))
        self.assertTrue(r.shape[1] == len(cols))
        # same column names
        self.assertSetEqual(set(r.columns.tolist()), set(cols))
    
    def test_impute_zero(self):
        series = self.df['age']
        msk_na = pd.isnull(series)
        impute = notation([Select('age'), Impute(0)])
        r = impute.fit_transform(self.df)
        

        n_notzero = (r.loc[msk_na, 'age'] != 0).sum()
        self.assertTrue(n_notzero==0)

    def test_keepothers(self):
        sel = notation((Select('ID')))
        r = sel.fit_transform(self.df)
        self.assertEqual(len(r.columns.tolist()), 1)

        sel = notation((Select('ID'), KeepOthers()))
        r = sel.fit_transform(self.df)
        all_columns = self.df.columns.tolist()
        self.assertSetEqual(set(all_columns), set(r.columns.tolist()))

    def test_notation(self):
        pl1 = notation([Impute(0)])
        self.assertTrue(isinstance(pl1, Pipeline))

        pl2 = notation([[[Impute(0)]]])
        self.assertTrue(isinstance(pl2, Pipeline))
        self.assertTrue(isinstance(pl2[0], Pipeline))
        self.assertTrue(isinstance(pl2[0][0], Pipeline))

        pl3 = notation(([Impute(0)], Select('age')))
        self.assertTrue(isinstance(pl3, PipelineUnion))
        self.assertTrue(isinstance(pl3[0], Pipeline))

    def test_Scale(self):
        col = 'height(cm)'; min_ = 0; max_ = 1
        self.assertGreater(self.df[col].max(), max_)
        ppl = notation([Select(col), Scale(min_, max_)])
        r = ppl.fit_transform(self.df)
        self.assertEqual(r[col].max(), max_)
        self.assertEqual(r[col].min(), min_)

    def test_winsorize(self):
        col = 'a'
        # by hard caps
        df = pd.DataFrame({col: list(range(1,101))})
        win = Winsorize(0, 10)
        r = win.fit_transform(df)
        self.assertLessEqual(df[col].max(), 10)
        self.assertGreaterEqual(df[col].min(), 0)

        # by percentiles
        df = pd.DataFrame({col: list(range(1,101))})
        win = Winsorize(0.05, 0.05)
        r = win.fit_transform(df)
        self.assertEqual(df[col].max(), 95)
        self.assertEqual((df[col]==95).sum(), 6)
        self.assertEqual(df[col].min(), 6)
        self.assertEqual((df[col]==6).sum(), 6)

    def test_make_dummy_ignore_unseen_values_in_transform(self):
        df1 = pd.DataFrame({'sex': ['M', 'F', 'M', 'M', 'M']})
        df2 = pd.DataFrame({'sex': ['F', 'X']})
        dummy = MakeDummy()
        r1 = dummy.fit_transform(df1)

        r1_cols = set(r1.columns.tolist())
        expected_r1_cols = {'sex_M', 'sex_F'}
        self.assertSetEqual(r1_cols, expected_r1_cols)

        r2 = dummy.transform(df2)

        r2_cols = set(r2.columns.tolist())
        self.assertSetEqual(r2_cols, expected_r1_cols)

    def test_drop_columns(self):
        drop_cols = ['sex', 'ID']
        drop_cols = notation((
            [Select(drop_cols), Drop()],
            KeepOthers()
            ))
        r = drop_cols.fit_transform(self.df)
        r_cols = r.columns.tolist()
        for col in drop_cols:
            self.assertTrue( col not in r_cols )
