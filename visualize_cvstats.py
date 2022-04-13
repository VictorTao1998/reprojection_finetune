import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import io
import plotly.express as px
import pandas as pd
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

pred_disp_path = "/media/jianyu/dataset/eval/cost_vol_visual/03_19_2022_23_03_37_/pred_disp_abs_err_cmap"
pred_cv_path = "/media/jianyu/dataset/eval/cost_vol_visual/03_19_2022_23_03_37_/cost_vol_pcd"
#gt_disp_path = "/media/jianyu/dataset/eval/psm_depth/vanilla/03_26_2022_20_32_36_/gt_disp"
prefix = "0-300002-0"
width = 480
height = 640
isdisp = True
if isdisp:
    ndisp = 192
else:
    ndisp = 160

fig, ax = plt.subplots()

#gt_disp = np.array(Image.open(os.path.join(gt_disp_path, prefix+'.png')), formats="L")
#print(gt_disp.shape)
disp_image = plt.imread(os.path.join(pred_disp_path, prefix + '.png'))
cv_np = np.load(os.path.join(pred_cv_path, prefix + '-data.npy'))
#print(cv_np.shape)
#fig, ax = plt.subplots()
sc = ax.imshow(disp_image)
mask = cv_np[:,200,300] > 1e-4
if isdisp:
    xa = np.array(list(range(ndisp)))
    xtext = 'disp'
else:
    xa = np.arange(0.01,1.61,0.01)
    xtext = 'depth'
x = xa[mask]
plt.plot(x,cv_np[:,200,300][mask])
img_buf = io.BytesIO()
plt.savefig(img_buf, format='png')

img = np.array(Image.open(img_buf))

imagebox = OffsetImage(img, zoom=0.5)
imagebox.image.axes = ax

annot = AnnotationBbox(imagebox, xy=(0,0), xybox=(100,200),
                        xycoords="data", boxcoords="offset points", pad=0.5,
                        arrowprops=dict( arrowstyle="->", connectionstyle="arc3,rad=-0.1"))
annot.set_visible(False)
ax.add_artist(annot)


#sc plt.scatter(x_embedded[:, 0], x_embedded[:, 1], c=mnist_int/10.0, cmap=cmap, s=3)

def update_annot(ind):
    #i = ind["ind"][0]
    pos = np.rint(ind).astype(int)#sc.get_offsets()[i]
    annot.xy = (pos[0], pos[1])
    #gt = gt_disp[pos[1], pos[0]]
    #print(gt)
    if pos[0] < 480:
        annot.xybox = (200, annot.xybox[1])
    else:
        annot.xybox = (-200, annot.xybox[1])

    if pos[1] > 270:
        annot.xybox = (annot.xybox[0],200)
    else:
        annot.xybox = (annot.xybox[0],-200)

    mask = cv_np[:,pos[1],pos[0]] > 1e-4

    x = xa[mask]
    y = cv_np[:,pos[1],pos[0]][mask]
    
    z = {xtext:x, "prob":y}
    f = pd.DataFrame(z)
    fig = px.line(f, x=xtext, y="prob")
    #fig.add_scatter(x=[gt], y=[0])
    img_buf = io.BytesIO()
    fig.write_image(img_buf)

    img = np.array(Image.open(img_buf))

    #img = mnist_img[i, :].reshape((width, height))
    imagebox.set_data(img)

def hover(event):
    vis = annot.get_visible()
    if event.inaxes == ax:
        output = sc.get_cursor_data(event)
        
        cont, ind = sc.contains(event)
        if cont:
            update_annot([event.xdata, event.ydata])
            annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()

fig.canvas.mpl_connect("motion_notify_event", hover)

plt.show()