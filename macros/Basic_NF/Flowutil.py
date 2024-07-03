'''
NORMALIZING FLOWS
'''
import torch
class Latent_data:
    def __init__(self, in_tensor,labels, sidebands = False, num_sample_features = 71, double = False):
        self.data = in_tensor
        if(double):
            self.double()
        self.labels = labels
        self.num_events = self.data.size()[0]
        self.latent_size = self.data.size()[1]
        self.num_sample_features = num_sample_features
    def get_sidebands(self, cut = 1.14):
        for i in range(len(self.data)):
            if(self.mass[i] < 1.14):
                self.data[i] = (9999 * torch.ones_like(self.data[i]))
                self.labels[i] = (9999 * torch.ones_like(self.labels[i]))
                self.mass[i] = (9999 * torch.ones_like(self.mass[i]))
        self.data = self.data[self.data[:,0] != 9999]
        self.labels = self.labels[self.labels[:,0] != 9999]
        self.mass = self.mass[self.mass[:] != 9999]
        self.num_events = self.data.size()[0]
    def set_batch_size(self,batch_size):
        self.batch_size = batch_size
        self.max_iter = int(self.num_events / self.batch_size)
    def set_mass(self, mass):
        self.mass = mass
    def sample(self,iteration = 0, random = False, _give_labels = False):
        if(random):
            return self.sample_random(give_labels = _give_labels)
        else:
            return self.sample_fixed(iteration,give_labels = _give_labels)
    def sample_fixed(self,iteration,give_labels = False):
        #0 index iterations - the "first" iteration is with iteration = 0
        # Calculate the first index we want to take from training data (rest of data is directly after)
        begin = iteration * self.batch_size
        # initialize
        samples = torch.zeros(self.batch_size, self.latent_size)
        labels = torch.zeros(self.batch_size, 1)
#         print(f"labels max (inside sample_fixed): {labels.max()}")
        #loop over consecutive tensors, save to return tensor
        if(give_labels):
            for i in range(self.batch_size):
                samples[i] = self.data[begin + i]
                labels[i] = self.labels[begin+i]
            return samples,labels
        else:
            for i in range(self.batch_size):
                samples[i] = self.data[begin + i]
            return samples
    def sample_random(self,labels = False):
        indices = rng.integers(low=0, high=self.num_events, size=self.batch_size)
        samples = torch.zeros(self.batch_size,self.latent_size)
        for index in range(len(indices)):
            samples[index] = self.data[indices[index]]
        return samples
    def double(self):
        self.data = torch.cat([self.data,self.data],dim=1)
        
def get_masked_affine(num_layers = 32, latent_dim = 71, hidden_dim = None, alternate_mask = True, switch_mask = True):
    if(hidden_dim == None):
        hidden_dim = latent_dim * 2
    #mask
    b = torch.ones(latent_dim)
#     print(f"len of b: {len(b)}")
    for i in range(b.size()[0]):
        if(alternate_mask):
            if i % 2 == 0:
                b[i] = 0
        else:
            if i // (latent_dim * 0.5) == 0:
                b[i] = 0
    masked_affine_flows = []
    for i in range(num_layers):
        s = nf.nets.MLP([latent_dim, hidden_dim, hidden_dim, latent_dim])
        t = nf.nets.MLP([latent_dim, hidden_dim,hidden_dim, latent_dim])
        if(switch_mask):
            if i % 2 == 0:
                masked_affine_flows += [nf.flows.MaskedAffineFlow(b, t, s)]
            else:
                masked_affine_flows += [nf.flows.MaskedAffineFlow(1 - b, t, s)]
        else:
            masked_affine_flows += [nf.flows.MaskedAffineFlow(1 - b, t, s)]
    return masked_affine_flows

def transform(in_data, model, reverse = True, distorted = False, show_progress = True):
    data_tensor = torch.zeros_like(in_data.data)
    model.eval()
    with torch.no_grad():
        with tqdm(total=in_data.max_iter, position=0, leave=True) as pbar:
            for it in range(in_data.max_iter):
                if(distorted):
                    test_samples = in_data.sample(iteration = it, distorted = True)
                else:
                    test_samples = in_data.sample(iteration = it)
                test_samples = test_samples.to(device)
                if(reverse):
                    output_batch = model.inverse(test_samples)
                else:
                    output_batch = model.forward(test_samples)
                for i in range(in_data.batch_size):
                    data_tensor[it*in_data.batch_size + i] = output_batch[i]
                if(show_progress):
                    pbar.update(1)
        if((in_data.max_iter * in_data.batch_size) != in_data.num_events):
            num_missing = in_data.num_events - (in_data.max_iter * in_data.batch_size)
            if(distorted):
                end_samples = in_data.distorted_features[(in_data.max_iter * in_data.batch_size):]
            else:
                end_samples = in_data.data[(in_data.max_iter * in_data.batch_size):]
            end_samples = end_samples.to(device)
            if(reverse):
                end_batch = model.inverse(end_samples)
            else:
                end_batch = model.forward(end_samples)
            for i in range(len(end_batch)):
                data_tensor[(in_data.max_iter * in_data.batch_size) + i] = end_batch[i]
    return data_tensor


def train(in_data, model, val = False,val_data = Latent_data(torch.empty(10000,71), torch.empty(10000,71)), num_epochs = 1, compact_num = 20, distorted = False, show_progress = True, lr = 5e-4):
    # train the MC model
    if(val):
        val_data.set_batch_size(int(np.floor(val_data.num_events / in_data.max_iter)))
        val_loss_hist = np.array([])
        full_val_loss_hist = np.array([])
    model.train()
    loss_hist = np.array([])
    full_loss_hist = np.array([])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    for i in range(num_epochs):
        with tqdm(total=in_data.max_iter, position=0, leave=True) as pbar:
            for it in range(in_data.max_iter):
                model.train()
                optimizer.zero_grad()
                #randomly sample the latent space
                if(distorted):
                    print(f"entered distorted")
                    samples = in_data.sample(iteration = it, distorted = True)
                else:
                    samples = in_data.sample(iteration = it)
                samples = samples.to(device)
                loss = model.forward_kld(samples)
                # Do backprop and optimizer step
                if ~(torch.isnan(loss) | torch.isinf(loss)):
                    loss.backward()
                    optimizer.step()
                # Log loss
                if~(torch.isnan(loss)):
                    full_loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())
                    if(loss < 1000):
                        loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())
                if(val):
                    model.eval()
                    val_samples = val_data.sample(iteration = it)
                    val_samples = val_samples.to(device)
                    val_loss = model.forward_kld(val_samples)
                    if~(torch.isnan(val_loss)):
                        full_val_loss_hist = np.append(val_loss_hist, val_loss.to('cpu').data.numpy())
                        if(val_loss < 1000):
                            val_loss_hist = np.append(val_loss_hist, val_loss.to('cpu').data.numpy())
                if(show_progress):
                    pbar.update(1)
                            
    #
    # This section of code exists solely to create more readable histograms
    #
    running_ttl = 0
    compact_hist = np.array([])
    j = 0
    for i in range(loss_hist.size):
        if(j != (i // compact_num)):
            compact_hist = np.append(compact_hist,running_ttl / compact_num)
            running_ttl = 0
        j = i // compact_num
        running_ttl += loss_hist[i]
    if(val):
        running_ttl_val = 0
        compact_hist_val = np.array([])
        j = 0
        for i in range(val_loss_hist.size):
            if(j != (i // compact_num)):
                compact_hist_val = np.append(compact_hist_val,running_ttl_val / compact_num)
                running_ttl_val = 0
            j = i // compact_num
            running_ttl_val += val_loss_hist[i]
        return compact_hist, compact_hist_val, full_loss_hist, full_val_loss_hist
    else:
        return compact_hist, loss_hist
    
def test(in_data, model, data_type = "none", distorted = False, return_loss = False,show_progress = True):
    model.eval()
    test_loss = 0
    counted_batches = 0
    with torch.no_grad():
        with tqdm(total=in_data.max_iter, position=0, leave=True) as pbar:
            for it in range(in_data.max_iter):
                if(distorted):
                    test_samples = in_data.sample(iteration = it, distorted = True)
                else:
                    test_samples = in_data.sample(iteration = it)
                test_samples = test_samples.to(device)
                new_loss = model.forward_kld(test_samples)
                if(not math.isnan(new_loss)):
                    test_loss += new_loss
                    counted_batches += 1
                if(show_progress):
                    pbar.update(1)
        if(data_type == "none"):
            print(f"average loss: {test_loss/counted_batches}")
        else:
            print(f"{data_type} average loss: {test_loss/counted_batches}")
    if(return_loss):
        return test_loss/counted_batches