
import pytwoway as tw
import bipartitepandas as bpd
import pandas as pd
import dask.dataframe as dd

# CRE
cre_params = tw.cre_params({'ndraw_trace': 50})
## Clustering ##
# Use firm-level cdfs of income as our measure
measures = bpd.measures.CDFs()
# Group using k-means
grouping = bpd.grouping.KMeans()
# General clustering
cluster_params = bpd.cluster_params({'measures': measures, 'grouping': grouping})
# Cleaning
clean_params_pre_collapse = bpd.clean_params({'connectedness': None, 'drop_returns': 'returners', 'copy': False})
clean_params_post_collapse = bpd.clean_params({'connectedness': 'leave_out_observation', 'drop_returns': False, 'is_sorted': True, 'copy': False})
# Simulating
#sim_params = bpd.sim_params({'n_workers': 1000, 'firm_size': 5, 'alpha_sig': 2, 'w_sig': 2, 'c_sort': 1.5, 'c_netw': 1.5, 'p_move': 0.1})

#sim_data = bpd.SimBipartite(sim_params).simulate()
sim_data = pd.read_csv('CRE_sim_data.csv')

# Convert into BipartitePandas DataFrame
bdf = bpd.BipartiteDataFrame(sim_data)
# Clean and collapse
bdf = bdf.clean(clean_params_pre_collapse).collapse(is_sorted=True, copy=False).clean(clean_params_post_collapse)
# Cluster
bdf = bdf.cluster(cluster_params)
# Convert to cross-section format
bdf_cs = bdf.to_eventstudy(is_sorted=True, copy=False).get_cs(copy=False)

# Initialize CRE estimator
cre_estimator = tw.CREEstimator(bdf_cs, cre_params)
# Fit CRE estimator
cre_estimator.fit()


cre_estimator.summary
