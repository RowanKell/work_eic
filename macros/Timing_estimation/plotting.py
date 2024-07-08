import matplotlib.pyplot as plot
import torch
test_data = torch.load("/cwork/rck32/eic/work_eic/macros/Timing_estimation/data/test/fixed_p_uniform.pt")
samples = torch.load("/cwork/rck32/eic/work_eic/macros/Timing_estimation/data/samples/fixed_p_uniform.pt")

sample_fig, (sample_axs,truth_axs) = plot.subplots(1,2,figsize=(12,5))
sample_fig.suptitle("Vary p cos theta distributions")
sample_axs.hist(samples,bins = 300)
sample_axs.set_title("learned distribution")
sample_axs.set_xlabel("time (ns)")
sample_axs.set_ylabel("counts")

truth_axs.hist(test_data[:,num_context],bins = 300)
truth_axs.set_title("truth distribution")
truth_axs.set_xlabel("time (ns)")
truth_axs.set_ylabel("counts")
sample_fig.show()

sample_fig.savefig("/cwork/rck32/eic/work_eic/macros/Timing_estimation/plots/test_distributions/fixed_p_uniform_full.pdf")