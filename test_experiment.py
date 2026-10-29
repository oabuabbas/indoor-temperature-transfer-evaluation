import unittest
import numpy as np
import pandas as pd
from adaptation_stress import fit, predict, adapt, quantile, SOURCE_END, ADAPT_START, CAL_START, CAL_END
from audit_nest import TEST_START, TEST_END
from multiseason import augment_features, augmented_fit, augmented_predict, SEASONS, CAL_END as SECOND_CAL_END
from horizons import HORIZONS
from external_pleia import forecast_for_sensor, F as EXTERNAL_FEATURES
from external_adaptation import make_sensor, model_delta
from daily_residual import select_weight,pval,FEATURES,T0,T1,T2,CAL0,CAL1
from prospective_gate import scores,apply_gate
from temporal_windows import window
from weather_forecast_sensitivity import lag_weather

class ExperimentChecks(unittest.TestCase):
    def test_windows_order_and_no_overlap(self):
        self.assertEqual(SOURCE_END, ADAPT_START)
        self.assertLess(ADAPT_START + pd.Timedelta(days=30), CAL_START)
        self.assertLess(CAL_END, TEST_START)
        self.assertLess(TEST_START, TEST_END)

    def test_intercept_and_feature_scale(self):
        x=np.column_stack([np.arange(40.),np.ones(40)])
        y=np.full(40,3.)
        model=fit(x,y)
        np.testing.assert_allclose(predict(model,x),y,atol=1e-10)
        self.assertTrue(np.isfinite(model[1]).all())

    def test_adaptation_returns_finite_predictions(self):
        x=np.arange(40.,dtype=float).reshape(20,2)
        model=fit(x,np.full(20,1.))
        adapted=adapt(model,x,np.full(20,2.))
        self.assertTrue(np.isfinite(predict(adapted,x)).all())

    def test_conformal_finite_sample_index(self):
        self.assertTrue(np.isnan(quantile(np.arange(23.))))
        self.assertEqual(quantile(np.arange(24.)),22.)

    def test_sensor_mask_drops_observed_value(self):
        x=np.array([[1.,2.,3.,4.,5.,6.],[1.,2.,999.,4.,5.,6.]])
        z=augment_features(x,np.zeros(6),np.ones(6),'missing_ambient')
        np.testing.assert_allclose(z[0],z[1])
        self.assertEqual(z[0,-2],1.)
        self.assertEqual(z[0,-1],0.)

    def test_multiseason_dates_and_augmented_fit(self):
        self.assertEqual(SECOND_CAL_END,SEASONS['winter'][0])
        self.assertEqual(SEASONS['winter'][1],SEASONS['spring'][0])
        x=np.random.default_rng(9).normal(size=(40,6))
        model=augmented_fit(x,np.ones(40))
        np.testing.assert_allclose(augmented_predict(model,x,'missing_solar'),1.,atol=1e-8)

    def test_forecast_horizons_are_forward_and_unique(self):
        self.assertEqual(HORIZONS,[1,6,24])
        self.assertTrue(all(h>0 for h in HORIZONS))
        self.assertEqual(len(set(HORIZONS)),len(HORIZONS))

    def test_external_forecast_target_and_missing_hour(self):
        t=pd.date_range('2021-02-01',periods=12,freq='h',tz='UTC')
        group=pd.DataFrame({'hour':t,'temp':np.arange(12,dtype=float)+20})
        d=forecast_for_sensor(group,6)
        first=d.iloc[0]
        self.assertEqual(first['target']-first['temp_now'],6.)
        self.assertNotIn('target',EXTERNAL_FEATURES)
        broken=group.drop(index=7)
        out=forecast_for_sensor(broken,6)
        self.assertNotIn(t[1],out.index)

    def test_external_output_matched_cases(self):
        from audit_nest import OUT
        path=OUT/'pleia_external_results.csv'
        if not path.exists(): self.skipTest('run external_pleia.py first')
        d=pd.read_csv(path)
        self.assertEqual(int((d.groupby(['horizon_h','sensor']).test_n.nunique()>1).sum()),0)
        self.assertEqual(int(d.duplicated(['horizon_h','sensor','model']).sum()),0)

    def test_daily_seasonal_reference_never_uses_future(self):
        t=pd.date_range('2021-02-01',periods=80,freq='h',tz='UTC')
        group=pd.DataFrame({'hour':t,'temp':np.arange(80,dtype=float)})
        for h in (1,6,24):
            f=make_sensor(group,h)
            issue=t[30]
            expected=group.set_index('hour').loc[issue-pd.Timedelta(hours=24-h),'temp']
            self.assertEqual(f.loc[issue,'seasonal_yesterday'],expected)
            if h==24:
                self.assertEqual(model_delta('daily_seasonal',None,f)[0],0.)

    def test_blend_selection_and_noop(self):
        x=np.arange(200.,dtype=float).reshape(40,5)
        model=fit(x,np.ones(40))
        d=pd.DataFrame(x,columns=FEATURES)
        d['target']=d['seasonal_yesterday']
        self.assertEqual(select_weight(model,d),0.)
        np.testing.assert_allclose(pval('blend_selected',model,0.,d),d['seasonal_yesterday'])

    def test_daily_residual_window_order(self):
        self.assertLess(T0,T1);self.assertLess(T1,T2);self.assertLess(T2,CAL0);self.assertLess(CAL0,CAL1)

    def test_ornl_output_comparisons_are_paired(self):
        from audit_nest import OUT
        path=OUT/'ornl_external_results.csv'
        if not path.exists(): self.skipTest('run external_ornl.py first')
        d=pd.read_csv(path)
        self.assertEqual(int((d.groupby(['horizon_h','case','room']).test_n.nunique()>1).sum()),0)
        self.assertEqual(int(d.duplicated(['horizon_h','case','room','model']).sum()),0)

    def test_gate_uses_input_scores_and_falls_back(self):
        d=pd.DataFrame({'temp_now':[20.,22.],'temp_lag1':[20.,20.],'seasonal_yesterday':[19.,21.],'target':[999.,-999.]})
        preds={'persistence':np.array([20.,22.]),'daily':np.array([19.,21.]),'adapted':np.array([20.5,22.5])}
        measured=scores(d,preds)
        out,chosen=apply_gate({'type':'threshold','fallback':'daily','score':'recent_change','threshold':1.},preds,measured)
        np.testing.assert_allclose(out,[20.5,21.]);np.testing.assert_array_equal(chosen,[True,False])
        d.target=[-999.,999.]
        for key,value in measured.items(): np.testing.assert_array_equal(value,scores(d,preds)[key])

    def test_gate_results_have_matched_samples(self):
        from audit_nest import OUT
        path=OUT/'prospective_gate_results.csv'
        if not path.exists(): self.skipTest('run prospective_gate.py first')
        d=pd.read_csv(path)
        self.assertEqual(int((d.groupby(['horizon_h','sensor','period']).test_n.nunique()>1).sum()),0)

    def test_issue_and_target_windows_for_all_horizons(self):
        boundary=pd.Timestamp('2021-03-01',tz='UTC')
        idx=pd.date_range(boundary-pd.Timedelta(hours=25),boundary+pd.Timedelta(hours=27),freq='h')
        data=pd.DataFrame({'x':np.arange(len(idx))},index=idx)
        for h in (1,6,24):
            selected=window(data,h,boundary,boundary+pd.Timedelta(days=2),boundary)
            self.assertTrue((selected.index>=boundary).all())
            self.assertTrue(((selected.index+pd.Timedelta(hours=h))>=boundary).all())
            self.assertNotIn(boundary-pd.Timedelta(hours=1),selected.index)

    def test_calibration_issue_after_adaptation(self):
        boundary=pd.Timestamp('2021-02-01',tz='UTC')
        cutoff=pd.Timestamp('2021-01-31',tz='UTC')
        idx=pd.date_range(cutoff-pd.Timedelta(hours=24),boundary+pd.Timedelta(hours=48),freq='h')
        data=pd.DataFrame({'x':np.arange(len(idx))},index=idx)
        result=window(data,24,boundary,boundary+pd.Timedelta(days=2),cutoff)
        self.assertTrue((result.index>=cutoff).all())
        self.assertTrue(((result.index+pd.Timedelta(hours=24))>=boundary).all())

    def test_weather_lag_uses_previous_observed_hour(self):
        t=pd.date_range('2021-01-01',periods=3,freq='h',tz='UTC')
        w=pd.DataFrame({'ambient':[10.,11.,12.],'solar':[0.,100.,200.]},index=t)
        lagged=lag_weather(w)
        self.assertEqual(lagged.loc[t[1],'ambient_lag1'],10.)
        self.assertEqual(lagged.loc[t[2],'solar_lag1'],100.)
        self.assertNotIn(t[0],lagged.index)

    def test_weather_sensitivity_matched_samples(self):
        from audit_nest import OUT
        path=OUT/'weather_forecast_sensitivity.csv'
        if not path.exists(): self.skipTest('run weather_forecast_sensitivity.py first')
        d=pd.read_csv(path)
        self.assertEqual(int((d.groupby(['horizon_h','sensor']).test_n.nunique()>1).sum()),0)
        self.assertEqual(int((d.groupby(['horizon_h','sensor']).cal_n.nunique()>1).sum()),0)

if __name__ == '__main__': unittest.main()
