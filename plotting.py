import argparse
import pandas as pd 
import matplotlib.pyplot as plt
import os


def plotting_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=str)
    return parser


parser = plotting_parser()
args = parser.parse_args()


metrics = df = pd.read_csv("Metric Evolution/" + args.model + ".csv")
epochs = {i: 5*(i+1) for i in range(len(metrics))}
metrics.rename(index=epochs, inplace=True)




if not os.path.exists("Metric Plots"): os.mkdir("Metric Plots")
if not os.path.exists("Metric Plots/" + args.model): os.mkdir("Metric Plots/" + args.model)


#Plot Entropy
entropy_data = metrics[[col for col in metrics if col.startswith('entropy dim')]]
plot = entropy_data.plot(figsize=(5, 3), legend=False).get_figure()
plt.xlabel("epcohs")
plt.ylabel("entropy")
plot.savefig('Metric Plots/' + args.model + '/entropy.png', dpi=300, bbox_inches='tight')
plot.clf()

#Plot Mu
mu_data = metrics[[col for col in metrics if col.startswith('mu dim')]]
plot = mu_data.plot(figsize=(5, 3), legend=False).get_figure()
plt.xlabel("epcohs")
plt.ylabel("mu")
plot.savefig('Metric Plots/' + args.model + '/mu.png', dpi=300, bbox_inches='tight')
plot.clf()

#Plot logvar or mu
mu_logvar_data = metrics[[col for col in metrics if col.startswith('logvar of mu dim')]]
plot = mu_logvar_data.plot(figsize=(6, 4), legend=False).get_figure()
plt.xlabel("epcohs")
plt.ylabel("logvar of sampled latent vectors")
plot.savefig('Metric Plots/' + args.model + '/logvar_of_mu.png', dpi=300, bbox_inches='tight')
plot.clf()

#Plot logvar
logvar_data = metrics[[col for col in metrics if col.startswith('logvar dim')]]
plot = logvar_data.plot(figsize=(5, 3), legend=False).get_figure()
plt.xlabel("epcohs")
plt.ylabel("predicted logvar")
plot.savefig('Metric Plots/' + args.model + '/logvar.pdf', dpi=300, bbox_inches='tight')
plot.clf()

#Plot perfect recon rate
perf_recon_data = metrics["perf recon rate"]
plot = perf_recon_data.plot(figsize=(5, 3), legend=False).get_figure()
plt.xlabel("epcohs")
plt.ylabel("prefect recon rate")
plot.savefig('Metric Plots/' + args.model + '/perf_recon.png', dpi=300, bbox_inches='tight')
plot.clf()

#Plot recon rate
recon_rate_data = metrics["recon rate"]
plot = recon_rate_data.plot(figsize=(5, 3), legend=False).get_figure()
plt.xlabel("epcohs")
plt.ylabel("recon rate")
plot.savefig('Metric Plots/' + args.model + '/recon_rate.png', dpi=300, bbox_inches='tight')
plot.clf()

#Plot perfect prefix length
perf_prefix_len_data = metrics["perf prefix len"]
plot = perf_prefix_len_data.plot(figsize=(5, 3), legend=False).get_figure()
plt.xlabel("epcohs")
plt.ylabel("perfect prefic len")
plot.savefig('Metric Plots/' + args.model + '/perf_pref_len.png', dpi=300, bbox_inches='tight')
plot.clf()

#Plot validity
validity_data = metrics["validity"]
plot = validity_data.plot(figsize=(5, 3), legend=False).get_figure()
plt.xlabel("epcohs")
plt.ylabel("validity")
plot.savefig('Metric Plots/' + args.model + '/validity.png', dpi=300, bbox_inches='tight')
plot.clf()

#plot stability
stability_data = metrics["stability"]
plot = stability_data.plot(figsize=(5, 3), legend=False).get_figure()
plt.xlabel("epcohs")
plt.ylabel("validity")
plot.savefig('Metric Plots/' + args.model + '/stability.png', dpi=300, bbox_inches='tight')
plot.clf()


#plot token_loss
token_loss_data = metrics[ "token loss"]
plot = token_loss_data.plot(figsize=(5, 3), legend=False).get_figure()
plt.xlabel("epcohs")
plt.ylabel("token loss")
plot.savefig('Metric Plots/' + args.model + '/token_loss.png', dpi=300, bbox_inches='tight')
plot.clf()

#Plot predicted length
length_data = metrics["length"]
plot = length_data.plot(figsize=(5, 3), legend=False).get_figure()
plt.xlabel("epcohs")
plt.ylabel("Predicted SELFIE length")
plot.savefig('Metric Plots/' + args.model + '/pred_length.png', dpi=300, bbox_inches='tight')
plot.clf()