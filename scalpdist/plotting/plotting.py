"""
Function for creating PDF output with analysis result.
"""

import matplotlib.backends.backend_pdf as pdf
import matplotlib.gridspec as gridspec

def plot_results(t1_img,brain_edge,head_edge,fname,spheres=None,dist_result=None,coords_dist=None):
    # Ensure masks are integers
    brain_edge = brain_edge.astype(int)
    head_edge = head_edge.astype(int)
    # Slices to show
    x_slices = np.round(np.array((t1_img.shape[0]/5,2*t1_img.shape[0]/5,t1_img.shape[0]/2,
                3*t1_img.shape[0]/5,4*t1_img.shape[0]/5)),0).astype(int)
    out_pdf = pdf.PdfPages(fname)
    fig = plt.figure(constrained_layout=True,figsize=(11,8))
    gs = fig.add_gridspec(3,5)
    for i,slice in enumerate(x_slices):
        ax = fig.add_subplot(gs[0,i])
        ax.imshow(ndimage.rotate(t1_img[slice,:,:],90),cmap='gray')
        ax.imshow(np.ma.masked_where(ndimage.rotate(head_edge[slice,:,:],90)==0,
                    ndimage.rotate(head_edge[slice,::],90)),cmap='viridis')
        ax.imshow(np.ma.masked_where(ndimage.rotate(brain_edge[slice,:,:],90)==0,
                    ndimage.rotate(brain_edge[slice,::],90)),cmap='viridis_r')
        ax.tick_params(axis='both',which='both',bottom=False,labelbottom=False,
                        left=False,labelleft=False)
    if not spheres is None:
        if isinstance(spheres,list):
            cnt = 0
            for i,sphere in enumerate(spheres):
                ax = fig.add_subplot(gs[1,cnt])
                ax.imshow(ndimage.rotate(t1_img[coords_dist[i][0],:,:],90),cmap='gray')
                ax.imshow(np.ma.masked_where(ndimage.rotate(sphere[coords_dist[i][0],:,:],90)==0,
                        ndimage.rotate(sphere[coords_dist[i][0],:,:],90)),cmap='viridis_r')
                ax.tick_params(axis='both',which='both',bottom=False,labelbottom=False,
                                left=False,labelleft=False)
                ax = fig.add_subplot(gs[1,cnt+1])
                ax.imshow(ndimage.rotate(t1_img[:,coords_dist[i][1],:],90),cmap='gray')
                ax.imshow(np.ma.masked_where(ndimage.rotate(sphere[:,coords_dist[i][1],:],90)==0,
                        ndimage.rotate(sphere[:,coords_dist[i][1],:],90)),cmap='viridis_r')
                ax.tick_params(axis='both',which='both',bottom=False,labelbottom=False,
                                left=False,labelleft=False)
                ax = fig.add_subplot(gs[2,cnt:cnt+2])
                ax.text(0.1,0.9,f'Region {i+1}',fontsize=14)
                plusminus = "\u00B1"
                ax.text(0.1,0.8,f"Distance = {dist_result[i][0]:0.1f} {plusminus} {dist_result[i][1]:0.1f} mm",fontsize=14)
                ax.axis('off')
                cnt = cnt+2
        else:
            ax = fig.add_subplot(gs[1,0])
            ax.imshow(ndimage.rotate(t1_img[coords_dist[0],:,:],90),cmap='gray')
            ax.imshow(np.ma.masked_where(ndimage.rotate(spheres[coords_dist[0],:,:],90)==0,
                        ndimage.rotate(spheres[coords_dist[0],::],90)),cmap='viridis_r')
            ax.tick_params(axis='both',which='both',bottom=False,labelbottom=False,
                        left=False,labelleft=False)
            ax = fig.add_subplot(gs[1,1])
            ax.imshow(ndimage.rotate(t1_img[:,coords_dist[1],:],90),cmap='gray')
            ax.imshow(np.ma.masked_where(ndimage.rotate(spheres[:,coords_dist[1],:],90)==0,
                        ndimage.rotate(spheres[:,coords_dist[1],:],90)),cmap='viridis_r')
            ax.tick_params(axis='both',which='both',bottom=False,labelbottom=False,
                        left=False,labelleft=False)
            ax = fig.add_subplot(gs[2,0:2])
            ax.text(0.1,0.9,'Region 1',fontsize=14)
            plusminus = "\u00B1"
            ax.text(0.1,0.8,f"Distance = {dist_result[0]:0.1f} {plusminus} {dist_result[1]:0.1f} mm",fontsize=14)
            ax.axis('off')
    out_pdf.savefig()
    out_pdf.close()
    plt.close()
