
# coding: utf-8

# In[ ]:


import nglview


def show_trajectory(traj, radiusScale=0.3):
    view = nglview.show_asetraj(traj)
    view.add_unitcell()
    view.add_spacefill()
    view.remove_ball_and_stick()
    view.camera = 'orthographic'
    view.parameters = {"clipDist": 0}
    view.center()
    view.update_spacefill(radiusType='covalent',
                          radiusScale=radiusScale,
                          color_scale='rainbow')
    return view

