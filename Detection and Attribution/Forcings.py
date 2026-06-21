import numpy as np
import matplotlib.pyplot as plt
import torch
import dsm_with_stein

F, a, b, c, s = 0.0, 1.0, 0.0, 1.0, 0.5
dt = 0.05
N_real = 310
n_ens = 1000
n_steps = 100
eps = 0.06
eps_vals = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]

np.random.seed(42)
torch.manual_seed(42)


#long trajectory and analytical score
N_long = 500_000
x = 0.0
traj = np.zeros(N_long)
score_exact = np.zeros(N_long)
for i in range(N_long):
    x += (F + a*x + b*x**2 - c*x**3)*dt + s*np.sqrt(dt)*np.random.randn()
    traj[i] = x
    score_exact[i] = 2*(F + a*x + b*x**2 - c*x**3) / s**2

#multiple means for Y_a(t)
mu1 = traj.mean()
mu2 = ((traj-mu1)**2).mean()
mu3 = ((traj-mu1)**3).mean()

traj_thin = traj[::200]

# learned score
data_tensor = torch.tensor(traj, dtype=torch.float32).view(-1, 1)
score_model, _ = dsm_with_stein.train(data_tensor, lambda_max=0.7, epochs=3000, batch_size=1024, K=4)
score_model.eval()
with torch.no_grad():
    score_dsm = score_model(data_tensor).numpy().flatten()
# score_dsm = stein_calibrate_score(traj, score_dsm)

#Response kernels (no Gaussian)
#state-independent forcings -- B(x) = -s(x)
R_exact = np.zeros((3, n_steps))
R_dsm = np.zeros((3, n_steps))
observables = [traj, (traj-mu1)**2, (traj-mu1)**3]
for m in range(3):
    for tau in range(n_steps):
        psi = observables[m]
        R_exact[m, tau] = -np.mean(score_exact[:N_long-tau]*psi[tau:])
        R_dsm[m, tau] = -np.mean(score_dsm[:N_long-tau]*psi[tau:])

#covariance of residual internal variability
d = 3 * n_steps
C_hat = np.zeros((d, d))
y_responses = []
for i in range(N_real):
    idx = np.random.randint(0, len(traj_thin)-n_ens)
    x_0 = traj_thin[idx:idx+n_ens].copy()
    x_a = x_0.copy()
    x_b = x_0.copy()
    Y_noise = np.zeros((3, n_steps))
    for t in range(n_steps):
        noise_a = s*np.sqrt(dt)*np.random.randn(n_ens)
        noise_b = s*np.sqrt(dt)*np.random.randn(n_ens)
        x_a += (F + a*x_a + b*x_a**2 - c*x_a**3)*dt + noise_a
        x_b += (F + a*x_b + b*x_b**2 - c*x_b**3)*dt + noise_b
        Y_noise[0, t] = x_a.mean() - x_b.mean()
        Y_noise[1, t] = ((x_a-mu1)**2).mean() - ((x_b-mu1)**2).mean()
        Y_noise[2, t] = ((x_a-mu1)**3).mean() - ((x_b-mu1)**3).mean()
    y_responses.append(np.concatenate([Y_noise[0], Y_noise[1], Y_noise[2]]))
y_responses = np.array(y_responses)
mu_Y = y_responses.mean(axis=0)
for i in range(N_real):
    C_hat += np.outer(y_responses[i] - mu_Y, y_responses[i] - mu_Y)
C_hat /= (N_real-1)
lam_shrink = 1e-2
C_hat = (1-lam_shrink)*C_hat + lam_shrink*np.trace(C_hat)/d*np.eye(d)
C_inv = np.linalg.inv(C_hat)

#forcing functions (to be recovered)
time_scale = np.arange(n_steps) * dt
g1 = np.ones(n_steps)
g2 = time_scale/ time_scale[-1]

#perturbed runs (N_real to smooth data)
Y_all = np.zeros((N_real, d))
for i in range(N_real):
    idx = np.random.randint(0, len(traj_thin) - n_ens)
    x0 = traj_thin[idx:idx+n_ens].copy()
    xu = x0.copy()
    xp = x0.copy()
    Y = np.zeros((3, n_steps))
    for t in range(n_steps):
        noise = s*np.sqrt(dt)*np.random.randn(n_ens)
        forcing = eps*g1[t] + eps*g2[t]   # both forcings at once
        xu += (F + a*xu + b*xu**2 - c*xu**3)*dt + noise
        xp += (F + a*xp + b*xp**2 - c*xp**3 + forcing)*dt + noise
        Y[0, t] = xp.mean() - xu.mean()
        Y[1, t] = ((xp-mu1)**2).mean() - ((xu-mu1)**2).mean()
        Y[2, t] = ((xp-mu1)**3).mean() - ((xu-mu1)**3).mean()
    Y_all[i] = np.concatenate([Y[0], Y[1], Y[2]])
Y = Y_all.mean(axis=0)
#block-Toeplitz operator
def build_operator(R):
    K = np.zeros((3*n_steps, n_steps))
    for m in range(3):              
        for t in range(n_steps):    
            for j in range(t+1):    
                K[m*n_steps + t, j] = dt * R[m, t-j]   #lag is t-j
    return K
K_exact = build_operator(R_exact)
K_dsm = build_operator(R_dsm)

#discrete-time derivative operator
D_t = np.zeros((n_steps, n_steps))          
for n in range(1, n_steps):
    D_t[n, n]   =  1.0
    D_t[n, n-1] = -1.0

def deconvolve(K, Y, lam):
    lhs   = K.T @ C_inv @ K + lam * (D_t.T @ D_t)   
    rhs = K.T @ C_inv @ Y                            
    return np.linalg.solve(lhs, rhs)   

lam_t = 10
g_exact = deconvolve(K_exact, Y, lam_t)
g_dsm = deconvolve(K_dsm, Y, lam_t)

#plot
g_true = eps * (g1 + g2)   #use whatever amplitude you put in the perturbed run
plt.plot(time_scale, g_true,  'k',  label='true')
plt.plot(time_scale, g_exact, 'r--', label='recovered (exact)')
plt.plot(time_scale, g_dsm,   'b--', label='recovered (DSM)')
plt.legend(); plt.show()
