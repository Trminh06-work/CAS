import torch
#mon -1,0,1 when model == 1
#mon 1:up up, 2 down down, 3 down up  4 up down, when model ==2
#   GPU Friendly  Lipschitz interpolation and approximation of multivariate scattered data
#   Handles increments in data at runtime, monotonicity constraints in each variable
#   No need for training
#   All calculations are performed on a specified device (GPU) via torch
#
#   LipFit provides various methods to interpolate/fit the multivariate scattered date, under the condition of Lipschitz continuity with Lipschitz constant LC (largest slope)
#   The data is first added (x and y values) and copied to GPU
#   Then values at query points are approximated by a Lipschitz function that fits the input data, using values(). Supply a guesstimate of Lipschitz constant. 
#   It can also be calculated exactly from data using LC=lipschitz_constant() (large data set it may be expensive, then use lipschitz_anchor_sampling instead)
#
#   It can enforce additional monotonicity constraints (set vector mon to 1,-1 or 0 for increasing/decreasing/neither and use setparams()) , each variable/feature is treated independently
#   mon=[1,-1,0] (think of a function f=x-y+z^2) . use model=1 in values()
#
#   It can enforce monotonicity on regions only. Set region boundaries (vectors a,b) and use setparams(). Then specify desired monotonicity. Here the meaning of mon vector components
#   is 1: up nothing up  /--/ (x<a increasing, a<x<b neither, b<x increasing, for each variable independently). 2: down nothing down \ -- \, 3: down nothing up \--/, 4 up nothing down /--\
#    use model=2 in values()  (a<=b). We then can have mon=[1,0,4,2] 
#
#   Function can be locally Lipschitz (ie LC depends on the region), first use compute_local_lipschitz() for a given data set, then   values_local() - will be a "smoother" interpolant
#   same rules apply for monotonicity and regions
#
#    Data smothing: if LC is underestimated, no Lipschitz finction with that LC can match the data exactly. The data will be smoothened to comply with given LC
#    In all above keep parameter k=1 
#    when k>1, need to set vector w (length k) like w=[0.8,0.1,0.1] and k=3 (values add to 1). Then LipFit will use more than closest neighbours and combine their weighted values with w  
#    (so nearest with 0.8 and the next two nearest with 0.1 weight each). The rest of the rules are the same as above. It is a different way of smoothing the data 
#
#
# Important methods:
#    initialisation
#    add (newdata)
#    values(query, type)
#    setparams(monotonicity, ranges, weights for knn type)
#    lipschitz_constant
#    compute_local_lipschitz
#    values_local
#
#
class LibLipT:
    def __init__(self, capacity, dim, knn, device):
        # Creates an empty instance, dimension dim and capacity, knn typically 1 but can be greater
        self.device=device
        if isinstance(knn, int)  and knn >=1:
            self.k=knn
        else: 
            self.k=1
        self.X = torch.empty((capacity, dim), device=self.device)
        self.Y = torch.empty((capacity, ), device=self.device)

        self.size = 0
        self.LipSplineX = torch.empty((1, 10), device=self.device)
        self.LipSplineY = torch.empty((1, 10), device=self.device)
        self.w = torch.full((self.k,), 1.0/self.k,device=self.device) #in case default value
        self.sampled_distances = torch.empty(self.size*6 , device=device)
        
    def add(self, X_new, Y_new):
        # adds new data to the dataset. Increases capacity if needed. For LocalLipschitz method, it has to be recalculated
        if not torch.is_tensor(X_new):
            X_new=torch.as_tensor(X_new).float()
        if not torch.is_tensor(Y_new):
            Y_new=torch.as_tensor(Y_new).float()         
            
        k = X_new.shape[0]        
        self.X = self.ensure_capacity(self.X,self.size,k)
        self.Y = self.ensure_capacity(self.Y,self.size,k)
        self.X[self.size:self.size+k].copy_(X_new)
        self.Y[self.size:self.size+k].copy_(Y_new)
        self.size += k
        
    def clear(self):    
        self.size=0
        
    def values(self, Q, M, model, k):
   # calculates a bunch of values at query points Q using the data added to this class
   # M is supposed Lipschitz constant, k is the number of neigbours (actually 2k)
   # typemodel can be 0,1,2 (normal, monotone, monotone on regions, the latter two requrie setparams). if k>1 also need to setparam
        k = self.k if k is None or k < -2 else k
        model =  0 if model is None  else model
            
        if k>1 and self.w.shape[0] < k:
            padded = torch.full((k,),int('0'),  dtype=self.w.dtype,   device=self.device )     
            padded[:self.w.shape[0]] = self.w
            self.w = padded          
                
                
                
        chunk_size = 1024
        temp=[]
 
        if not torch.is_tensor(Q):
            Q = torch.as_tensor(Q).float()
        if Q.ndim == 1:
            if self.X.shape[1]==1:
                Q=Q.unsqueeze(1) #many 1dim samples
            else:    
                Q=Q.unsqueeze(0) #one sample
            
        for i in range(0, Q.shape[0], chunk_size):
            x_chunk = Q[i:i+chunk_size].to(self.device)
            
            out_chunk = self.value_device(x_chunk,M, model,k)
           
            # bring result back if needed
            out_chunk = out_chunk.cpu()   
            temp.append(out_chunk)
            
        return torch.cat(temp, dim=0)
            
    def setparams(self,mon,a,b,w):
        # Sets monotonicity types and regions, and also weights for knn-type calculations
        
        target_len = self.X.shape[1]
        
        if a is not None:
            a=torch.as_tensor(a).float()
            if a.ndim != 1:
                a=a.flatten()
            if a.shape[0] < target_len:
                padded = torch.full((target_len,),float('-inf'),  dtype=a.dtype,   device=a.device )     
                padded[:a.shape[0]] = a
                a = padded
            self.a = torch.as_tensor(a, device=self.device)
        else: 
            self.a = torch.full((self.X.shape[1],), -torch.inf, device=self.device)
        
        if b is not None:
            b=torch.as_tensor(b).float()
            if b.ndim != 1:
                b=b.flatten().float()
            if b.shape[0] < target_len:
                padded = torch.full((target_len,),float('inf'),  dtype=b.dtype,   device=b.device )     
                padded[:b.shape[0]] = b
                b = padded
            self.b = torch.as_tensor(b, device=self.device)
        else: 
            self.b = torch.full((self.X.shape[1],), torch.inf, device=self.device)  
            
        if mon is not None:
            mon=torch.as_tensor(mon).float()
            if mon.ndim != 1:
                mon=mon.flatten()
            if mon.shape[0] < target_len:
                padded = torch.zeros(target_len, dtype=mon.dtype, device=mon.device)
                padded[:mon.shape[0]] = mon
                mon = padded           
            self.mon = torch.as_tensor(mon, device=self.device)
        else:
            self.mon = torch.full((self.X.shape[1],),0,device=self.device)    

        if w is not None:
            w = torch.as_tensor(w).float()
            self.w =  torch.as_tensor(w, device=self.device)
        else: 
            self.w = torch.full((self.k,), 1.0/self.k,device=self.device)     
            
  


    def lipschitz_constant(self, block_size=1024, eps=1e-10):
       # calculates the Lipschitz constant of the data set, complexity n^2
        """
        X: (M, d)
        y: (M, 1) or (M,)
        returns: scalar Lipschitz constant
        """
        M = self.size
        if self.Y.ndim == 2:
           Yview = self.Y.squeeze(-1)
           Yview=Yview[:M] 
        else:
            Yview=self.Y[:M]

        Xview=self.X[:M]

        max_ratio = torch.tensor(0.0, device=self.device)

        for i in range(0, M, block_size):
            Xi = Xview[i:i+block_size]              # (Bi, d)
            yi = Yview[i:i+block_size]              # (Bi,)

            for j in range(0, M, block_size):
                Xj = Xview[j:j+block_size]          # (Bj, d)
                yj = Yview[j:j+block_size]          # (Bj,)

            # Pairwise differences
                dx = Xi[:, None, :] - Xj[None, :, :]     # (Bi, Bj, d)
                dy = yi[:, None] - yj[None, :]           # (Bi, Bj)

                dist = torch.linalg.norm(dx, dim=-1)     # (Bi, Bj)

            # Avoid division by zero (diagonal or duplicate points)
                dist = torch.clamp(dist, min=eps)
                ratios = torch.zeros_like(dist)
                ratios = torch.abs(dy) / dist

                max_ratio = torch.maximum(max_ratio, ratios.max())

        return max_ratio.item()	
        
    def lipschitz_anchor_sampling(self, num_anchors=512, samples_per_anchor=512, eps=1e-12):
        #calculates Lipschitz constant of the data using a sample of anchor points, less expensive
        if self.Y.ndim == 2:
            self.Y = self.Y.squeeze(-1)
        M = self.size
        max_ratio = torch.tensor(0.0, device=self.device)
        anchors = torch.randint(0, M, (num_anchors,), device=self.device)
        for a in anchors:
            Xa = self.X[a]                      # (d,)
            ya = self.Y[a]

            j = torch.randint(0, M, (samples_per_anchor,), device=self.device)

            Xj = self.X[j]
            yj = self.Y[j]

            dx = Xj - Xa
            dy = yj - ya

            dist = torch.linalg.norm(dx, dim=-1)
            mask = dist > eps

            ratios = torch.zeros_like(dist)
            ratios[mask] = torch.abs(dy[mask]) / dist[mask]

            max_ratio = torch.maximum(max_ratio, ratios.max())

        return max_ratio.item()            


    def compute_local_lipschitz(self, S=16):
        # computes local Lipschitz constants for every datum, approximated by a spline (S knots)
        # has to be called whenever new data is added
        X_view = self.X[:self.size]
        Y_view = self.Y[:self.size]   
        self.LipSplineX = torch.empty((self.size, S), device=self.device)
        self.LipSplineY = torch.empty((self.size, S), device=self.device)
        self.sampled_distances = torch.empty(self.size*10 , device=self.device)
        offset=0
        
        for n in range(self.size):
            Q = X_view[n]
            dist = torch.cdist(Q.unsqueeze(0), X_view).clamp_min_(1e-8)
            yi=Y_view[n]
            dy = torch.abs(yi - Y_view)        # (size,)
            Lip = dy / (dist +1e-12)  # can be /(D+1e-8)
            Lip = Lip.flatten()
            #Lip[n]=0
            if n<10 or self.size<1000:
                xs_out, ys_out, n_valid	 = self.select_monotone_pairs_spread( dist, Lip, S)
                dist=dist.flatten()
                if n<10:
                    self.sampled_distances[offset:offset+dist.shape[0]] = dist
                    offset += dist.shape[0]
            else:
                xs_out, ys_out, n_valid	 = self.select_monotone_pairs_spread1( dist, Lip, S, q_bins)  #or 2
                
            if n==9 and self.size>=1000: # last one    
                q = torch.linspace(0, 1, S + 1, device=self.device)
                q_bins  = torch.quantile(self.sampled_distances, q)
#                print(q_bins)
#                print(self.sampled_distances)
                
            self.LipSplineX[n,:].copy_(xs_out)
            self.LipSplineY[n,:].copy_(ys_out)

                
        
    def values_local(self, Q, model, k):
        # after computing local Lipschitz, calculates values at a bunch of queries Q.  
        # k is the number of neigbours (actually 2k)
        # typemodel can be 0,1,2 (normal, monotone, monotone on regions, the latter two requrie setparams). if k>1 also need to setparam
        k = self.k if k is None or k < -2 else k
        model =  0 if model is None  else model
        
        if k>1 and self.w.shape[0] < k:
            padded = torch.full((k,), int('0'),  dtype=self.w.dtype,   device=self.device )     
            padded[:self.w.shape[0]] = self.w
            self.w = padded           
            
            
        # sanity check and defaults 
        if self.LipSplineX.shape[0] < self.size or  self.LipSplineY.shape[0] < self.size:
            return self.values(Q,1,model,k)
        
        chunk_size = 1024
        temp=[]
        if not torch.is_tensor(Q):
            Q = torch.as_tensor(Q, dtype=torch.float32)
        if Q.ndim == 1:
            if self.X.shape[1]==1:
                Q=Q.unsqueeze(1) #many 1dim samples
            else:    
                Q=Q.unsqueeze(0) #one sample
                        
            
        for i in range(0, Q.shape[0], chunk_size):
            x_chunk = Q[i:i+chunk_size].to(self.device)
            
            out_chunk = self.value_device_local(x_chunk, model,k)
           
            # bring result back if needed
            out_chunk = out_chunk.cpu()   
            temp.append(out_chunk)
            
        return torch.cat(temp, dim=0)

# ==================== Internal private methods ==============================        
    def query_simple(self, M, Q, k):
        X_view = self.X[:self.size]
        Y_view = self.Y[:self.size]
        D = torch.cdist(Q.unsqueeze(0), X_view)
        
        UB = Y_view + M*D
        D = Y_view - M*D

        return torch.topk(UB, k, largest=False), torch.topk(D, k, largest=True)	 

    def query_mon(self, M, Q, k, mon):
        X_view = self.X[:self.size]
        Y_view = self.Y[:self.size]
        # here distances
        mon_view = mon[:self.X.shape[1]]
        DL, DR = self.mon_l2(Q, X_view, mon_view)
        
        DL  = Y_view + M*DL
        DR  = Y_view - M*DR
        
        return torch.topk(DL, k, largest=False), torch.topk(DR, k, largest=True)

    def query_mon_bound(self, M, Q, k, mon, a, b):
        X_view = self.X[:self.size]
        Y_view = self.Y[:self.size]
        # here distances
        
        DL, DR = self.mon_bounds_l2(Q, X_view, a[:self.X.shape[1]],b[:self.X.shape[1]], mon[:self.X.shape[1]])
        
        DL  = Y_view + M*DL
        DR  = Y_view - M*DR
        
        return torch.topk(DL, k, largest=False), torch.topk(DR, k, largest=True)        
        
    def ensure_capacity(self, X, size, new_k):
        if size + new_k > X.shape[0]:
            new_cap = max(X.shape[0] * 2, size + new_k)
            if X.dim()>1:
                X_new = torch.empty((new_cap, X.shape[1]), device=X.device)
            else:
                X_new = torch.empty((new_cap,), device=X.device)

            X_new[:size].copy_(X[:size])

            return X_new
        return X	        
        
    def mon_l2(self, q, x, mon):
        x = x
        s = torch.sign(mon)
        q = q.squeeze(0)  if q.dim() == 2 else q

        #L=upper
        d = q - x
        DL = torch.abs(d)
        DR = s * d

        w = torch.abs(s) 

       # L =  torch.where(s == 0, DL, torch.clamp(-DR, min=0))
       # R =  torch.where(s == 0, DL, torch.clamp(DR, min=0))
        L=(1-w)*DL + w * torch.relu(DR)
        R=(1-w)*DL + w * torch.relu(-DR)
        
        DL = torch.sqrt((L ** 2).sum(dim=-1))
        DR = torch.sqrt((R ** 2).sum(dim=-1))
        return DL, DR

    def mon_bounds_l2(self, q, x, a, b, montype):
        #montype 1:up up, 2 down down (can be -1), 3 down up  4 up down
        x = x
        q = q.squeeze(0)  if q.dim() == 2 else q
        
        ff1, ff2, ff1_t, ff2_t, ff4, ff5, absqx=   self.ff_all1(q, x, a, b)
        
        #upper
        L = torch.where(montype == 0, absqx**2, torch.where(montype == 1, ff1**2,
            torch.where((montype == 2) | (montype == -1), ff1_t**2,
            torch.where(montype == 3, ff4**2, ff5**2) )))
         
        #lower 
        R = torch.where(montype == 0, absqx**2, torch.where(montype == 1, ff2**2,
            torch.where((montype == 2) | (montype == -1), ff2_t**2,
            torch.where(montype == 3, ff5**2, ff4**2) ))  )           
        
        DL = torch.sqrt((L ).sum(dim=-1))
        DR = torch.sqrt((R ).sum(dim=-1))
        return DL, DR
        
        
# ===== calculations of distinct bounds for type 2 (partial monotonicity on regions ========
    def ff_all(self,x, xc, a, b):
        x_ = x.unsqueeze(0)
        a_ = a.unsqueeze(0)
        b_ = b.unsqueeze(0)

        def core(x_, xc, a_, b_):
            min_b_xc = torch.minimum(b_, xc)
            max_x_a  = torch.maximum(x_, a_)
            max_xc_a = torch.maximum(xc, a_)

            f1 = torch.clamp(min_b_xc - max_x_a, min=0) + torch.clamp(x_ - xc, min=0)
            f2 = -(torch.clamp(torch.minimum(b_, x_) - max_xc_a, min=0)
               + torch.clamp(xc - x_, min=0))
            return f1, f2

    # original
        f1, f2 = core(x_, xc, a_, b_)

    # transformed inputs
        f1t, f2t = core(-x_, -xc, -b_, -a_)

        return f1, f2, f1t, f2t    
        
    def ff_all1(self, x, xc, a, b):
        x_ = x.unsqueeze(0)
        a_ = a.unsqueeze(0)
        b_ = b.unsqueeze(0)

    # ---- shared primitives (original) ----
        min_b_xc = torch.minimum(b_, xc)
        max_x_a  = torch.maximum(x_, a_)
        max_x_b  = torch.maximum(x_, b_)
        max_xc_a = torch.maximum(xc, a_)
        min_b_x  = torch.minimum(b_, x_)
        min_a_x  = torch.minimum(a_, x_)
        max_b_xc = torch.maximum(b_, xc)
        abs_x_xc = torch.abs(x_-xc)

    # ---- ff1(x) ----
        ff1 = torch.clamp(min_b_xc - max_x_a, min=0) + torch.clamp(x_ - xc, min=0)

    # ---- ff2(x) ----
        ff2 = -(torch.clamp(min_b_x - max_xc_a, min=0)  + torch.clamp(xc - x_, min=0))
        
    # =========================================================
    # ff4
    # max(min(b,xc) - x, x - max(a,xc), 0)
    # =========================================================

       # ff4 = (
       #     abs_x_xc
       #     - torch.relu(min_a_x  - xc)
       #     - torch.relu(xc - max_x_b))
        ff4 = (
            abs_x_xc
             - torch.relu(min_a_x  - xc)
            +  torch.clamp(-xc + max_x_b,max=0))
    # =========================================================
    # ff5
    # min(x-xc, xc-x) + max(0, min(a,xc)-x) + max(0, x-max(b,xc))
    # =========================================================    
        ff5 = (   -abs_x_xc
            + torch.relu(torch.minimum(a_, xc) - x_)
            + torch.relu(x_ - max_b_xc))    

    # =========================================================
    # transformed case WITHOUT recomputing full min/max logic
    # =========================================================

    # ff1(-x, -xc, -b, -a)

    # min(-b, -xc) - max(-x, -a)
        ff1_t_part1 = torch.clamp(
            (-max_xc_a) - ( -min_b_x ), min=0  )

        ff1_t_part2 = torch.clamp((-x_) +xc, min=0)

        ff1_t = ff1_t_part1 + ff1_t_part2

    # ff2(-x, -xc, -b, -a)

        ff2_t_part1 = torch.clamp(
            (min_b_xc) - (max_x_a), min=0    )

        ff2_t_part2 = torch.clamp(x_ -xc, min=0)

        ff2_t = -(ff2_t_part1 + ff2_t_part2)

        return ff1, ff2, ff1_t, ff2_t, ff4, ff5, abs_x_xc 

# =========== internal calculations ========
    def OWA2(self, valsu, valsl, wei):
        return 0.5* ((valsu+valsl) * wei).sum()
        
    def value_device(self,Q,M,typemodel, k):
        # calculates the values at query points one by one
        out1= torch.empty((Q.shape[0], ), device=self.device)
        for i in range(Q.shape[0]):
            Q_view = Q[i]
            match typemodel:
                case 0:
                    (ub_vals, ub_idx), (lb_vals, lb_idx)  = self.query_simple(M,Q_view, k)
                case 1:
                    (ub_vals, ub_idx), (lb_vals, lb_idx) = self.query_mon(M,Q_view, max(k,1), self.mon)
                case 2:
                    (ub_vals, ub_idx), (lb_vals, lb_idx) = self.query_mon_bound(M, Q_view, max(k,1), self.mon, self.a, self.b)
                case _:
                    (ub_vals, ub_idx), (lb_vals, lb_idx)  = self.query_simple(M,Q_view, k)

            ub_vals=ub_vals.flatten()
            lb_vals=lb_vals.flatten()
            
            if k==1 :
                out1[i] = 0.5*(ub_vals[0]+lb_vals[0])
            elif k==2:
                out1[i] = 0.5*( self.w[0]*ub_vals[0] +self.w[1]*ub_vals[1] +   self.w[0]*lb_vals[0] + self.w[1]*lb_vals[1])
            elif k==-1:
                out1[i] = ub_vals[0]
            elif k==-2:
                out1[i] = lb_vals[0]
            else:
                out1[i] =self.OWA2(ub_vals,lb_vals,self.w[:k])
  
        return out1    

           
  # ========== for Local Lipschitz, building splines ========          
 
    def eval_splines_with_constant_tail(self, Tx, Ty, xq):
        x = Tx[:, :]  # (B, K)
        y = Ty[:, :]  # (B, K)

        B, K = x.shape
        device = Tx.device
        xq=xq.flatten()
        #xq = xq.unsqueeze(1)
        # find interval
        #i = torch.searchsorted(x, xq) - 1
        i = (torch.searchsorted(x, xq.unsqueeze(1)) - 1).squeeze(1)

        # detect out-of-bounds on the right
        right_mask = i >= (K - 1)
        i = torch.clamp(i, 0, K - 2)

        batch_idx = torch.arange(B, device=device)

        # normal interpolation
    #    x0 = x[batch_idx, i]
    #    x1 = x[batch_idx, i + 1]
    #    y0 = y[batch_idx, i]
        y1 = y[batch_idx, i + 1]

      #  t = (xq - x0) / (x1 - x0 + 1e-12)
      #  out = y0 + t * (y1 - y0)
        out=y1
        
    # ---- right tail mask ----
        right = i >= (K - 1)
        
        if right.any():
            out=y[batch_idx, K-1]

        return out


    def select_monotone_pairs_spread(self,    x,  y,  M   ):
        #constructs a piecewise constant spline approximating Lipschitz constant for each datum
        """
        x, y: (K,)

        Returns:
            xs_out: (M,)
            ys_out: (M,)
            n_valid: number before padding
        """

        device = x.device
        x = x.flatten()
        y = y.flatten()
        # sort by x
        idx = torch.argsort(x)
        xs = x[idx]
        ys = y[idx]

        xs = xs[1:]
        ys = ys[1:]
        # monotone filter
        running_max = torch.cummax(ys, dim=0).values

        keep = torch.empty_like(ys, device=device, dtype=torch.bool)
        keep[0] = True
        keep[1:] = ys[1:] > (running_max[:-1] )

        xs_sel = xs[keep]
        ys_sel = ys[keep]
        N = xs_sel.shape[0]
        # output tensors
        xs_out = torch.empty(M, device=device, dtype=x.dtype)
        ys_out = torch.empty(M, device=device, dtype=y.dtype)

        if N >= M:
            # evenly spaced indices
            #idx_spread = torch.arange(M) * N // M
            idx_spread = torch.round( torch.linspace(0, N - 1, M)).long()
            
            xs_out[:] = xs_sel[idx_spread]
            ys_out[:] = ys_sel[idx_spread]
                
            n_valid = M
            
        else:
            # copy all valid points
            xs_out[:N] = xs_sel
            ys_out[:N] = ys_sel

            # pad with last value
            xs_out[N:] = xs_sel[-1]+1
            ys_out[N:] = ys_sel[-1]
            
        #    yss = torch.zeros_like(xs_out)
        #    yss[1:] = torch.cumsum(ys_out[:-1] * (xs_out[1:] - xs_out[:-1]), dim=0)
            n_valid = N

        return xs_out, ys_out, n_valid	
        

    def value_device_local(self, Q,  typemodel, k):
        out1= torch.empty((Q.shape[0], ), device=self.device)
        for i in range(Q.shape[0]):
            Q_view = Q[i]
            match typemodel:
                case 0:
                    (ub_vals, ub_idx), (lb_vals, lb_idx)  = self.query_simple_local(Q_view, max(k,1))
                case 1:
                    (ub_vals, ub_idx), (lb_vals, lb_idx) = self.query_mon_local(Q_view, max(k,1), self.mon)
                case 2:
                    (ub_vals, ub_idx), (lb_vals, lb_idx) = self.query_mon_bound_local( Q_view, max(k,1), self.mon, self.a, self.b)
                case _:
                    (ub_vals, ub_idx), (lb_vals, lb_idx)  = self.query_simple_local(Q_view, max(k,1))
                    
            ub_vals=ub_vals.flatten()
            lb_vals=lb_vals.flatten()
            if k==1 :
                out1[i] = 0.5*(ub_vals[0]+lb_vals[0])
            elif k==2:
                out1[i] = 0.5*( self.w[0]*ub_vals[0] +self.w[1]*ub_vals[1] +   self.w[0]*lb_vals[0] + self.w[1]*lb_vals[1])
            elif k==-1:
                out1[i] = ub_vals[0]
            elif k==-2:
                out1[i] = lb_vals[0]
            else:
                out1[i] =self.OWA2(ub_vals,lb_vals,self.w[:k])
  
        return out1          
        
    def query_simple_local(self, Q, k):
        X_view = self.X[:self.size]
        Y_view = self.Y[:self.size]
        D = torch.cdist(Q.unsqueeze(0), X_view)
        
        M = self.eval_splines_with_constant_tail(self.LipSplineX, self.LipSplineY, D)
        UB = Y_view + M*D
        LB= Y_view - M*D

        return torch.topk(UB, k, largest=False), torch.topk(LB, k, largest=True)	    
        
    def query_mon_local(self,  Q, k, mon):
        X_view = self.X[:self.size]
        Y_view = self.Y[:self.size]
        # here distances
        mon_view = mon[:self.X.shape[1]]
        DL, DR = self.mon_l2(Q, X_view, mon_view)
        M = self.eval_splines_with_constant_tail(self.LipSplineX, self.LipSplineY, torch.max(DL,DR))
        
        DL  = Y_view + M*DL
        DR  = Y_view - M*DR
        
        return torch.topk(DL, k, largest=False), torch.topk(DR, k, largest=True)
        
    def query_mon_bound_local(self, Q, k, mon, a, b):
        X_view = self.X[:self.size]
        Y_view = self.Y[:self.size]
        # here distances
        
        DL, DR = self.mon_bounds_l2(Q, X_view, a[:self.X.shape[1]],b[:self.X.shape[1]], mon[:self.X.shape[1]])
        M = self.eval_splines_with_constant_tail(self.LipSplineX, self.LipSplineY, torch.max(DL,DR))
        
        DL  = Y_view + M*DL
        DR  = Y_view - M*DR
        
        return torch.topk(DL, k, largest=False), torch.topk(DR, k, largest=True)               
        
           
    def printme(self):
        print(self.X)
        print(self.LipSplineX)
        print(self.LipSplineY)

    def select_monotone_pairs_spread1(self,    x,  y,  M , q_bins  ):
        #constructs a piecewise constant spline approximating Lipschitz constant for each datum
        """
        x, y: (K,)

        Returns:
            xs_out: (M,)
            ys_out: (M,)
            n_valid: number before padding
        """

        device = x.device
        x = x.flatten()
        y = y.flatten()
        
        K=16 #number of blocks
        kn=4
        vals, _ = torch.topk(x, kn, largest=False)
        #define bins 
        vals = torch.sort(vals).values
        vals = vals[vals > 0]
        

        
        bins = torch.cat([q_bins, vals])
        bins = torch.unique(bins)
        bins = torch.sort(bins).values
        bins = torch.quantile(bins, torch.linspace(0, 1, M + 1, device=device))# resample
        
        bin_idx = torch.bucketize(x, bins, right=False)
        bin_idx = bin_idx.clamp(0, len(bins) - 2) 
        
        rep_id  = torch.arange(x.shape[0], device=x.device) % K

        B=bins.shape[0]-1
        flat_idx = rep_id * B + bin_idx

        local = torch.full((K * B,), -torch.inf, device=device)

        local.scatter_reduce_(
            0,
            flat_idx,
            y,
            reduce="amax"
            )
            
        local = local.view(K, B)    
        binmax = local.max(dim=0).values
        env = torch.cummax(binmax, dim=0).values
        
        
        # output tensors
        xs_out = torch.empty(M, device=device, dtype=x.dtype)
        ys_out = torch.empty(M, device=device, dtype=y.dtype)
        
        N = env.shape[0]
        n_valid = M

        if N >= M:
            
            xs_out[:] = bins[:-1]
            ys_out[:] = env
                
            n_valid = M
            
        else:
            # copy all valid points
            xs_out[:N] = bins[:-1]
            ys_out[:N] = env

            # pad with last value
            xs_out[N:] = bins[-1]+1
            ys_out[N:] = env[-1]
            
        #    yss = torch.zeros_like(xs_out)
        #    yss[1:] = torch.cumsum(ys_out[:-1] * (xs_out[1:] - xs_out[:-1]), dim=0)
            

        return xs_out, ys_out, n_valid	    

    def select_monotone_pairs_spread2(self,    x,  y,  M , q_bins  ):
        #constructs a piecewise constant spline approximating Lipschitz constant for each datum
        """
        x, y: (K,)

        Returns:
            xs_out: (M,)
            ys_out: (M,)
            n_valid: number before padding
        """

        device = x.device
        x = x.flatten()
        y = y.flatten()
        
        kn=4
        vals, _ = torch.topk(x, kn, largest=False)
        #define bins 
        vals = torch.sort(vals).values
        vals = vals[vals > 0]
                
        bins = torch.cat([q_bins, vals])
        bins = torch.unique(bins)
        bins = torch.sort(bins).values
        bins = torch.quantile(bins, torch.linspace(0, 1, M + 1, device=device))# resample

        B=bins.shape[0]-1
        env = torch.empty(B, device=device)
        
        running_max = torch.tensor(-torch.inf, device=device)

        for b in range(B):

            left  = bins[b]
            right = bins[b + 1]

            mask = (x >= left) & (x < right)
            current = torch.where(mask, y, -torch.inf).max()
            running_max = torch.maximum(running_max, current)

            env[b] = running_max    
        
        # output tensors
        xs_out = torch.empty(M, device=device, dtype=x.dtype)
        ys_out = torch.empty(M, device=device, dtype=y.dtype)
        
        n_valid = env.shape[0]
            
        xs_out[:] = bins[:-1]
        ys_out[:] = env
                
        return xs_out, ys_out, n_valid	
    def select_monotone_pairs_spread3(self,    x,  y,  M , q_bins  ):
        #constructs a piecewise constant spline approximating Lipschitz constant for each datum
        """
        x, y: (K,)

        Returns:
            xs_out: (M,)
            ys_out: (M,)
            n_valid: number before padding
        """

        device = x.device
        x = x.flatten()
        y = y.flatten()
        
        kn=4
        vals, _ = torch.topk(x, kn, largest=False)
        #define bins 
        vals = torch.sort(vals).values
        vals = vals[vals > 0]
                
        bins = torch.cat([q_bins, vals])
        bins = torch.unique(bins)
        bins = torch.sort(bins).values
        bins = torch.quantile(bins, torch.linspace(0, 1, M + 1, device=device))# resample

        bin_idx = torch.bucketize(x, bins) - 1
        binmax = torch.full((bins.shape[0]-1,), -torch.inf, device=device)

        binmax.scatter_reduce_(
            0,
            bin_idx,
            y,
            reduce="amax",
            include_self=True
        )
        env = torch.cummax(binmax, dim=0).values 
        
        # output tensors
        xs_out = torch.empty(M, device=device, dtype=x.dtype)
        ys_out = torch.empty(M, device=device, dtype=y.dtype)
        
        n_valid = env.shape[0]
            
        xs_out[:] = bins[:-1]
        ys_out[:] = env
                
        return xs_out, ys_out, n_valid	        
        

        
        
# ===================== example of usage ==========================
if __name__ == "__main__":
    # sample usage code
    import torch
    import numpy as np
    def get_device():
        if torch.cuda.is_available():
            return torch.device("cuda") #ROCm as well
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    device = get_device()

    dim = 3
    LL =  LibLipT(1000,dim, 1, device)

    M, N = 100, dim
    x = torch.rand(M, N)
    y = torch.zeros(M)

    for i in range(0,M):
        y[i]=torch.sin(x[i,0])+0.5*torch.sin(5*x[i,1]) + x[i,2]
    
    LL.clear()    #only if removing previous data
    LL.add(x,y)


    mon=[1,1,-1]
    a=[0,0,-1]
    b=[1,1,2]
    w=[1,0,0,0,0]
    LL.setparams(mon,a,b,w)

    M1=100 #test data
    q = torch.rand(M1, N)
    yp=LL.values(q, 2.0, 0, 1) #guess Lip constant 2

    LC=LL.lipschitz_constant() #actually calculate it
    yp1=LL.values(q, LC, 0, 1)
    
    yt = torch.zeros(M1)
    for i in range(0,M1):
        yt[i]=torch.sin(q[i,0])+0.5*torch.sin(5*q[i,1]) + q[i,2]
    

#compare   yt and yp
    print(torch.abs(yp-yt))
    
# local Lipschitz    
    LL.compute_local_lipschitz(S=8)
    
    #yp=LL.values(xxx, 1, 0, 1)
    yp=LL.values_local(q, 0, 1)    
    
#plotting 1d
    xx = np.linspace(-3, 5.0, 300)
    xxx = torch.as_tensor(xx, dtype=torch.float32).unsqueeze(1)


    #plt.figure(figsize=(10, 6)) 

    #zv =  yp.numpy()
    #plt.plot(xx, zv, label="")
    #plt.plot(x, y, 'o')
    #plt.show()

#==========================================================
#  explanations
# 
# 1. get your GPU with device = get_device()
# 2. Instantiate LibLipT object LL =  LibLipT(Reserve, dim, 1, device), 
#    Reserve is how much space reserve for data (data can be added later with .add)
#    dimension of x, use knn=1 for now, as default value for interpolation
#
# 3.  Add data set (X and Y) as tensors of shape (M,dim), dim
# 4. set parameters: mon,a,b,w. For w use [1,0,0,0,..] as many values as your knn, until I tell you. Can be extra 0 beyond knn
#  mon is tensor (or just vector) of size dim, components can be -1,0,1,2,3,4  (for each dimension type of monotonicity)
#  a,b can be none, or vectors of size dim with this meaning: the coordinate domain is split into (-inf,a,b,inf)
#  it only matters when typemodel ==2 (see later) -1 decreasing 0 neither, 1 increasing (on (-inf,a) and (b,inf), 
#  in the interval(a,b) no monotonicity assumed at all. mon=2(same as -1, decreasing), mon=3 decreasing then increasing (~convex)
#   mon=4 increasing the decreasing (~concave). No monotonicity on (a,b)
# 
#   5. prepare test data (like q = torch.rand(M1, N))
#   6. calculate them yp=LL.values(q, Lipconst , typemon, knn) 
#     Lipconst = your educated guess, typemon can be 0,1,2 (0 nothing, 1 increasing/decreasing on the domain of each coordinate 
#    (set vector mon to -1,0, or 1), or 2 (then need to set a,b, and mon vector like 0,1,2,3,4 see above).
#  7. check the resulting values, compare with ground truth
# 
#  8. Now, LocalLipschitz works