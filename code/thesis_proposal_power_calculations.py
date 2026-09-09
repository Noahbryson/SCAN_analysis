from src.functions.stat_power_calculations import motor_mapping_power, thalamocortical_LMM_power
import pandas as pd


aim1Flag = True
aim3Flag = False
if aim1Flag:
      a1 = motor_mapping_power()
      pt_counts = [i for i in range(12,50,2)]
      print(pt_counts)
      n_spins = 1000
      variance_summary = a1.estimate_aim1_jackknife_variance(
            n_spins=n_spins,
      )
      print(variance_summary)
      alpha = 0.005
      result = a1.calculate_aim1_power_curve(
            variance_summary=variance_summary,
            patient_counts=pt_counts,
            alpha=alpha
      )
      print(result)
      # variance_summary.to_csv('aim1_jackknife_variance.csv')
      result.to_csv(f'aim1_jackknife_power_{alpha}-alpha.csv')
      print(0)
if aim3Flag:
      a3 = thalamocortical_LMM_power()
      aim3_effect_size_table = r"""
            \begin{table}[ht]
            \centering
            \begin{tabular}{lccc}
            \toprule
            Scenario & $\beta_5$ & $\beta_4$ & $\beta_7$ \\
            \midrule
            Conservative & 0.05 & 0.15 & 0.075 \\
            Expected & 0.10 & 0.30 & 0.20 \\
            Optimistic & 0.15 & 0.40 & 0.30 \\
            \bottomrule
            \end{tabular}
            \caption{Aim 3 effect-size assumptions used for power simulations. $\beta_5$ denotes the functional class by motor response interaction, $\beta_4$ denotes the functional class by cue-target coherence interaction, and $\beta_7$ denotes the three-way interaction.}
            \label{tab:aim3_effect_sizes}
            \end{table}
            """
      print(aim3_effect_size_table)
      pt_counts = [i for i in range(4,24,4)]
      print(pt_counts)
      motor_electrodes_per_pt = 8
      intereffector_electrodes_per_pt = 3
      total_trials = 200
      hit_rate = 0.70 #percent hit rate from task difficulty titration
      tp_trials = int(total_trials * 0.72 * hit_rate)
      fp_trials = int(total_trials * 0.18 * (1-hit_rate))
      tn_trials = int(total_trials * 0.18 * hit_rate)
      fn_trials = int(total_trials * 0.72 * (1-hit_rate))
      artifact_fraction = 0.05
      intercept = 1e-3
      # Conservative
      beta_func_motor = 0.05
      beta_func_coherence = 0.15
      beta_func_cohere_motor = 0.075

      res1 = a3.simulate_aim3_mixedlm_power(pt_counts,motor_electrodes_per_pt,intereffector_electrodes_per_pt, tp_trials,tn_trials,fp_trials,fn_trials,artifact_fraction,intercept,beta_functional_class_by_coherence=beta_func_coherence,beta_functional_class_by_motor_response=beta_func_motor,beta_functional_class_by_coherence_by_motor_response=beta_func_cohere_motor,return_simulation_summary=True)
      # Expected
      beta_func_motor = 0.10
      beta_func_coherence = 0.30
      beta_func_cohere_motor = 0.20

      res2 = a3.simulate_aim3_mixedlm_power(pt_counts,motor_electrodes_per_pt,intereffector_electrodes_per_pt, tp_trials,tn_trials,fp_trials,fn_trials,artifact_fraction,intercept,beta_functional_class_by_coherence=beta_func_coherence,beta_functional_class_by_motor_response=beta_func_motor,beta_functional_class_by_coherence_by_motor_response=beta_func_cohere_motor,return_simulation_summary=True)
      # Optimistic
      beta_func_motor = 0.15
      beta_func_coherence = 0.40
      beta_func_cohere_motor = 0.30

      res3 = a3.simulate_aim3_mixedlm_power(pt_counts,motor_electrodes_per_pt,intereffector_electrodes_per_pt, tp_trials,tn_trials,fp_trials,fn_trials,artifact_fraction,intercept,beta_functional_class_by_coherence=beta_func_coherence,beta_functional_class_by_motor_response=beta_func_motor,beta_functional_class_by_coherence_by_motor_response=beta_func_cohere_motor,return_simulation_summary=True)
      res1['confidence_class'] = 'conservative'
      print(res1)
      res2['confidence_class'] = 'expected'
      print(res2)
      res3['confidence_class'] = 'optimistic'
      print(res3)

      result = pd.concat([res1,res2,res3])

      result.to_csv('aim3_simulation.csv')
      print(0)
