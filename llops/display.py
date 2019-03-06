import numpy as np
from .base import abs, real, shape, ndim
from .mem import getMemoryUsage


def addScaleBar(pixel_size_um):
    import matplotlib.pyplot as plt
    from matplotlib_scalebar.scalebar import ScaleBar
    scalebar = ScaleBar(pixel_size_um, 'um')
    plt.gca().add_artist(scalebar)

def listPlotFlat(list_to_plot, labels="Item %d", ax=None, fig=None,
                 figsize=None, max_width=6, colorbar=False, clim=None,  **kwargs):
    import matplotlib.pyplot as plt
    import math

    # Determine aspect ratio
    if ndim(list_to_plot[0]) == 2:
        aspect_ratio = shape(list_to_plot[0])[0] / shape(list_to_plot[0])[1]
    else:
        aspect_ratio = 1.0

    # Determine number of rows and columns
    col_count = round(min(len(list_to_plot) * aspect_ratio, max_width))
    row_count = math.ceil(len(list_to_plot) / col_count)

    # Calculate optimal figsize
    if fig is None:
        fig = plt.figure()

        if figsize is not None:
            _figsize = list(figsize)
        else:
            figsize = [12, 3]
            figsize[1] *= row_count / aspect_ratio

    # Parse title
    if labels:
        if type(labels) not in (list, tuple):
            labels = [labels] * len(list_to_plot)

    # Generate plot
    for index, value in enumerate(list_to_plot):
        plt.subplot(row_count, col_count, index + 1)
        if ndim(value) == 2:
            plt.imshow(real(value), **kwargs)

            # Turn off axis labels
            plt.axis('off')
        elif ndim(value) == 1:
            plt.plot(real(value), **kwargs)
            plt.tight_layout()

        # Set colorbar (if provided)
        if colorbar:
            plt.colorbar()

        # Set colorbar limits (if provided)
        if clim:
            plt.clim(clim)

        # Set title (if provided)
        if labels:

            if '%d' in labels[index]:
                plt.title(labels[index] % index)
            else:
                plt.title(labels[index])


def listPlotScroll(list_to_plot, fig=None, colorbar=False, **kwargs):
    from matplotlib.widgets import Slider
    from matplotlib import pyplot as plt

    if fig is None:
        fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (6, 5)))

    plt.subplots_adjust(left=0.25, bottom=0.1)
    idx_0 = 0

    plt_handle = plt.imshow(np.abs(list_to_plot[0]), **kwargs)

    if colorbar:
        plt.colorbar()
    ax_slider = plt.axes([0.25, 0.01, 0.65, 0.03])
    slider = Slider(ax_slider, 'Image Index', 0, len(list_to_plot), valinit=idx_0, valfmt='%i')

    def update(val):
        idx = int(np.round(slider.val))
        plt_handle.set_data(np.abs(list_to_plot[idx]))
        fig.canvas.draw()

    slider.on_changed(update)
    plt.show()

    return slider


def objToString(object, text_color="", use_newline=True):
    '''
    Helper method for serialization of any object
    '''
    var_dict = vars(object)
    if use_newline:
        keys = list(var_dict.keys())
        keys.sort()
        str_to_return = '\n'.join(['%s: %s' % (Color.BOLD + text_color + key +
                                               Color.END, var_dict[key]) for key in reversed(keys)])
    else:
        keys = list(var_dict.keys())
        keys.sort()
        str_to_return = ', '.join(['%s: %s' % (Color.BOLD + text_color + key +
                                               Color.END, var_dict[key]) for key in reversed(keys)])

    return(str_to_return)


def saveAnimation(anim, output_file):
    import matplotlib.pyplot as plt
    from matplotlib import animation
    # Write movie
    if output_file is not None:
        from matplotlib import animation
        # Set up formatting for the movie files
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=7.5, bitrate=1600)

        # Save Animastion (with black background)
        anim.save(output_file, writer=writer)


def imshow(x, **kwargs):
    """Wrapper for imshow which takes the absolute value of the imput"""
    import matplotlib.pyplot as plt
    plt.imshow(abs(x), **kwargs)


def animatePositionFrameList(fig, position_list, frame_list, frame_title=None,
                             position_title=None, index_format=None, position_volume_size=None,
                             interval_ms=None, loop=False, position_units='um',
                             frame_units='um', title=None, parameter_list=None,
                             label_repr=None, pixel_size=None, position_angle=None,
                             label_values=None, **kwargs):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Parse titles
    position_title = 'Position' if position_title is None else position_title
    frame_title = 'Frame' if frame_title is None else frame_title
    title = 'State' if title is None else title
    index_format = '%g' if index_format is None else index_format

    # Create 3D axis
    ax0 = fig.add_subplot(121, projection='3d')
    ax1 = fig.add_subplot(122)

    # Parse loop behavior
    frame_indicies = list(range(len(position_list))) if not loop else list(range(len(position_list))) + list(reversed(range(len(position_list))))
    position_list = position_list if not loop else position_list + list(reversed(position_list))
    parameter_list = np.arange(len(frame_indicies)) if parameter_list is None else parameter_list

    # This funciton updates the point cloud
    def update_graph(index):
        data = position_list[index]
        frame_index = frame_indicies[index]
        graph.set_data(data[:,0], data[:,1])
        graph.set_3d_properties(data[:,2])
        im.set_data(frame_list[frame_index])
        _title.set_text(('%s: %s') % (title, index_format % tuple(np.asarray(parameter_list).T[index])))
        return _title_positions, _title_frame, graph,

    # Determine maximum extent in each dimension
    points_np = np.vstack(position_list)

    if position_volume_size is None:
        ax0.set_xlim((np.min(points_np[:, 0]), np.max(points_np[:, 0])))
        ax0.set_ylim((np.min(points_np[:, 1]), np.max(points_np[:, 1])))
        ax0.set_zlim((np.min(points_np[:, 2]), np.max(points_np[:, 2])))
    else:
        ax0.set_xlim((-position_volume_size[0] / 2, position_volume_size[0] / 2))
        ax0.set_ylim((-position_volume_size[1] / 2, position_volume_size[1] / 2))
        ax0.set_zlim((-position_volume_size[2], 0))

    # Set initial view angle of scatter plot
    if position_angle is not None:
        ax0.view_init(position_angle[0], position_angle[1])

    _title_positions = ax0.set_title(('%s') % (position_title))
    _title_frame = ax1.set_title(('%s') % (frame_title))

    _title = plt.suptitle(('%s: %s') % (title, index_format % tuple(np.asarray(parameter_list).T[0])))

    data = position_list[0]
    graph, = ax0.plot(data[:,0], data[:,1], data[:,2], linestyle="", marker=kwargs.pop('marker', 'o'),
                                                                     markersize=kwargs.pop('markersize', 5),
                                                                     markeredgewidth=kwargs.pop('markeredgewidth', 1),
                                                                     markeredgecolor=kwargs.pop('markeredgecolor', 'k'))

    ax0.set_xlabel('x (%s)' % position_units)
    ax0.set_ylabel('y (%s)' % position_units)
    ax0.set_zlabel('z (%s)' % position_units)

    im = ax1.imshow(frame_list[0])
    plt.axis('off')

    # Add scale bar if pixel size is supplied

    if pixel_size is not None:
        from matplotlib_scalebar.scalebar import ScaleBar
        scalebar = ScaleBar(pixel_size, frame_units) # 1 pixel = 0.2 1/cm
        ax1.add_artist(scalebar)

    plt.tight_layout()

    anim = animate(fig, update_graph, frame_indicies, interval=interval_ms, blit=True, **kwargs)

    return anim


def animatePositionList(fig, position_list, title='Positions', frame_indicies=None, interval_ms=None, loop=False, ax=None, **kwargs):
    import matplotlib.pyplot as plt

    # Create 3D axis
    ax = fig.add_subplot(111, projection='3d') if ax is None else ax

    # Parse loop behavior
    frame_indicies = list(range(len(position_list))) if not loop else list(range(len(position_list))) + list(reversed(range(len(position_list))))
    position_list = position_list if not loop else position_list + list(reversed(position_list))

    # This funciton updates the point cloud
    def update_graph(index):
        data = position_list[index]
        frame_index = frame_indicies[index]
        graph.set_data(data[:,0], data[:,1])
        graph.set_3d_properties(data[:,2])
        _title.set_text('{}, index={}'.format(title, frame_index))
        return _title, graph,

    # Determine maximum extent in each dimension
    points_np = np.vstack(position_list)
    ax.set_xlim((np.min(points_np[:, 0]), np.max(points_np[:, 0])))
    ax.set_ylim((np.min(points_np[:, 1]), np.max(points_np[:, 1])))
    ax.set_zlim((np.min(points_np[:, 2]), np.max(points_np[:, 2])))

    _title = ax.set_title(title)

    data = position_list[0]
    graph, = ax.plot(data[:,0], data[:,1], data[:,2], linestyle="", marker=kwargs.pop('marker', 'o'),
                                                                    markersize=kwargs.pop('markersize', 5),
                                                                    markeredgewidth=kwargs.pop('markeredgewidth', 1),
                                                                    markeredgecolor=kwargs.pop('markeredgecolor', 'k'))


    plt.tight_layout()

    anim = animate(fig, update_graph, frame_indicies, interval=interval_ms, blit=True, **kwargs)

    return anim

def animate(fig, func, frames, fargs=None, **kwargs):
    import matplotlib.animation as animation
    anim_running = True

    def onClick(event):
        nonlocal anim_running
        if anim_running:
            anim.event_source.stop()
            anim_running = False
        else:
            anim.event_source.start()
            anim_running = True

    fig.canvas.mpl_connect('button_press_event', onClick)

    anim = animation.FuncAnimation(fig, func, frames, fargs=fargs, **kwargs)
    return anim


def showImageStack(image_stack, figsize=None, caxis='full', **kwargs):
    from matplotlib.widgets import Slider, Button, RadioButtons
    from matplotlib import pyplot as plt

    image_stack = np.asarray(image_stack)
    fig, ax = plt.subplots(figsize=figsize)
    plt.subplots_adjust(left=0.25, bottom=0.1)
    idx_0 = 0

    max_val = np.max(image_stack)
    plt_handle = plt.imshow(np.abs(image_stack[idx_0, :, :]), **kwargs)
    plt.colorbar()
    ax_slider = plt.axes([0.25, 0.01, 0.65, 0.03])
    slider = Slider(ax_slider, 'Image Index', 0, image_stack.shape[0], valinit=idx_0, valfmt='%i')

    def update(val):
        idx = int(np.round(slider.val))
        plt_handle.set_data(np.abs(image_stack[idx]))
        fig.canvas.draw()

    slider.on_changed(update)
    plt.show()

    return slider


def progressBar(sequence, width=None, every=None, size=None, name='Items', text_color='white'):
    from ipywidgets import IntProgress, HTML, VBox
    from IPython.display import display

    is_iterator = False
    if size is None:
        try:
            size = len(sequence)
        except TypeError:
            is_iterator = True

    if size is not None:
        if every is None:
            if size <= 200:
                every = 1
            else:
                every = int(size / 200)     # every 0.5%
    else:
        assert every is not None, 'sequence is iterator, set every'

    if is_iterator:
        progress = IntProgress(min=0, max=1, value=0, width=width)
        progress.bar_style = 'info'
    else:
        progress = IntProgress(min=0, max=size, value=0, width=width)

    label = HTML()
    box = VBox(children=[label, progress])
    display(box)

    index = 0
    try:
        for index, record in enumerate(sequence, 0):
            if index == 0 or index % every == 0:
                if is_iterator:
                    label.value = '<font color="{color}">{name}: {index} / ?</font>'.format(
                    color=text_color,
                        name=name,
                        index=index
                    )
                else:
                    progress.value = index
                    label.value = u'<font color="{color}">{name}: {index} / {size}</font>'.format(
                        color=text_color,
                        name=name,
                        index=index,
                        size=size
                    )
            yield record
    except:
        progress.bar_style = 'danger'
        raise
    else:
        progress.bar_style = 'success'
        progress.value = index+1
        label.value = u'<font color="{color}">{name}: {index} / {size}</font>'.format(
            color=text_color,
            name=name,
            index=str(index+1 or '?'),
            size=size
        )

class IterationText():
    def __init__(self):
        print(Color.BLUE + "|  Iter  |      Cost      | Elapsed time (s) |  Norm of Step  | Memory Usage (CPU/GPU) |" + Color.END)
        print(Color.BLUE + "+ ------ + -------------- + ---------------- + -------------- + ---------------------- +" + Color.END)

    def update(self, iteration, new_cost, new_time=0, step_norm=0):
        memory_usage = getMemoryUsage()
        print("|% 5d   |    %.02e    |    % 7.2f       |    %.02e    | %6.1f MB / %6.1f MB  |" % (iteration, abs(new_cost), new_time, step_norm, memory_usage['cpu'], memory_usage['gpu']))


class IterationPlot():
    def __init__(self, ax, iteration_count, cost=None,
                 use_log=(False,False), **kwargs):

        self.ax = ax
        self.iteration_count = iteration_count
        if cost is not None:
            self.cost = cost
        else:
            self.cost = [0]

        index = 0
        if not any(use_log):
            self.plot, = ax.plot(index, self.cost[index], c='y', label='Cost', **kwargs)
        elif use_log[0] and not use_log[1]:
            self.plot, = ax.semilogx(index, self.cost[index], c='y', label='Cost', **kwargs)
        elif use_log[1] and not use_log[0]:
            self.plot, = ax.semilogy(index, self.cost[index], c='y', label='Cost', **kwargs)
        else:
            self.plot, = ax.loglog(index, self.cost[index], c='y', label='Cost', **kwargs)

        self.ax.set_xlim([1, iteration_count])
        self.ax.set_title(kwargs.pop('title', ''))
        self.ax.set_xlabel(kwargs.pop('x_label', 'Iteration'))
        self.ax.set_ylabel(kwargs.pop('y_label', 'Cost'))
        self.ax.legend()

    def update(self, iteration, new_cost, **kwargs):
        index = np.arange(iteration)
        self.cost.append(new_cost)
        self.plot.set_xdata(index)
        self.plot.set_ydata(self.cost[-1])
        self.ax.set_ylim((1e-10, max(self.cost)))

class Color:
    '''
    This class is used for print coloring
    '''
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
