import matplotlib.pyplot as plot
import torch
test_data = torch.load("/cwork/rck32/eic/work_eic/macros/Timing_estimation/data/test/vary_p_cos_theta.pt")
samples_cut = torch.load("/cwork/rck32/eic/work_eic/macros/Timing_estimation/data/samples/vary_p_cos_theta.pt")

sample_fig, (sample_axs,truth_axs) = plot.subplots(2,1,figsize=(5,10))
sample_fig.suptitle("Vary p cos theta distributions")
sample_axs.hist(samples_cut,bins = 100)
sample_axs.set_title("learned distribution")
sample_axs.set_xlabel("time (ns)")
sample_axs.set_ylabel("counts")

truth_axs.hist(test_data[:,num_context],bins = 100)
truth_axs.set_title("truth distribution")
truth_axs.set_xlabel("time (ns)")
truth_axs.set_ylabel("counts")
sample_fig.show()

sample_fig.savefig("/cwork/rck32/eic/work_eic/macros/Timing_estimation/plots/test_distributions/vary_p_cos_theta.pdf")