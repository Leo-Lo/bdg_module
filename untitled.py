



for (Δ,evals) in zip(Δs,eval_list):

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    dim1,dim2 = N,N
    X = np.arange(0,dim1,1)
    Y = np.arange(0,dim2,1)
    X, Y = np.meshgrid(X, Y)
    
    print('Δ='+str(Δ)+',μ='+str(μ))
    for Z in evals:
        Z = np.reshape(Z,(dim1,dim2))
        surf = ax.plot_surface(X, Y, Z*factor, cmap=cm.viridis,linewidth=0, antialiased=False)

    ax.view_init(elev=0, azim=59.)
    ax.set_zbound([-1e-3*factor,1e-3*factor])
#     ax.set_title('Δ='+str(Δ)+',μ='+str(μ),loc='right')
    ax.set_xticks([])
    ax.set_yticks([])
#     ax.set_zticks([-0.01*factor,-0.005*factor,0,0.005*factor,0.01*factor])
#     ax.set_zlabel('E (meV)')
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    
    for t in ax.zaxis.get_major_ticks(): t.label.set_fontsize(20)

#     plt.savefig('Data ('+name+')/3D plot (Δ='+str(Δ)+',μ='+str(μ)+').png',bbox_inches='tight',dpi=200)
    plt.show()