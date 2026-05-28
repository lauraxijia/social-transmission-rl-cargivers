import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import json
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import seaborn as sns
from matplotlib.font_manager import FontProperties

"""Helperfunctions for plots."""

def state_to_xy(world_matrix, state):
    """
    Converts a state number to (x, y) coordinates in an 8x8 grid.
    
    Args:
    - state: The state number.
    - grid_width: The width of the grid (should be 8 for an 8x8 grid).
    
    Returns:
    Tuple of (x, y) coordinates.
    """
    return np.where(world_matrix == state)


def add_boundaries(ax, world, boundaries, grid_size):
    """Adds boundaries to the plot."""
    if not boundaries:
        raise ValueError("boundaries must not be empty")
    
    for line in boundaries:
        
        if len(line) != 2:
            raise ValueError("each line must have exactly two elements")
        
        converted_lines = (state_to_xy(world, line[0]), state_to_xy(world, line[1]))
        horizontal = converted_lines[0][1] - converted_lines[1][1]
        vertical = converted_lines[0][0] - converted_lines[1][0]

        if vertical == 0:
            plot_vertical_boundary(ax, converted_lines, grid_size)
        else:
            plot_horizontal_boundary(ax, converted_lines, grid_size)

def plot_vertical_boundary(ax, converted_lines, grid_size):
    """Plots a vertical boundary."""
    x_max = np.max([converted_lines[0][1], converted_lines[1][1]])
    x_argmax = np.argmax([converted_lines[0][1], converted_lines[1][1]])

    if x_argmax == 0:
        y_axis = [grid_size-converted_lines[0][0], grid_size-converted_lines[1][0]-1]
    else: 
        y_axis = [grid_size-converted_lines[0][0]-1, grid_size-converted_lines[1][0]]
    ax.plot([x_max, x_max], y_axis, 'k-', linewidth=4, color="#464343ff")


def plot_horizontal_boundary(ax, converted_lines, grid_size):
    """Plots a horizontal boundary."""
    y_max = np.max([converted_lines[0][0], converted_lines[1][0]])
    ax.plot([converted_lines[0][1], converted_lines[0][1]+1] , [grid_size-y_max, grid_size-y_max], 'k-', linewidth=4, color="#464343ff")


# To get the last point of the arrow for tm plot
def get_end_center(direction, end, grid_size):
    """Returns the end center based on the direction."""
    if direction == "up":
        return (end[1] + 0.5, grid_size - end[0] - 0.25)
    elif direction == "right":
        return (end[1] + 0.75, grid_size - end[0] - 0.5)
    elif direction == "down":
        return (end[1] + 0.5, grid_size - end[0] - 0.75)
    else:  # direction == "left"
        return (end[1] + 0.25, grid_size - end[0] - 0.5)

# Plot arrows to illustrate tm
def draw_arrow(ax, start_center, end_center, color):
    """Draws an arrow from start center to end center."""
    ax.annotate("",
                xy=end_center, xycoords='data',
                xytext=start_center, textcoords='data',
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color=color))


def plot_world(env, world, reward_info, state_num = None, boundaries = None, reward_text = False, reward_color = False, dashed_squares = None, transition_matrix = None, exp = 'baseline', savefigpath = False, ax = None):
    """
    Plots the world grid with optional attributes
    env: Environment object
    world: 2D numpy array representing the grid world
    reward_info: 2D numpy array with the reward information 
    state_num: Boolean to display state numbers
    boundaries: List of tuples with the boundaries
    reward_text: Boolean to display reward values
    reward_color: Boolean to color the rewards
    dashed_squares: List of states to draw dashed squares
    transition_matrix: 3D numpy array with the transition matrix
    exp: String indicating the experiment type (e.g., 'baseline', 'exp2', 'exp3') to determine the start locations
    savefigpath: Path to save the figure
    ax: Matplotlib axis object

    Returns:
    fig: Matplotlib figure object
    """
    
    grid_size = world.shape[0]
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
        #ax = plt.gca()  # Get the current axis if none is provided
    
    
    # Create a blank 8x8 white grid
    ax.set_xlim([0, grid_size])
    ax.set_ylim([0, grid_size])
    ax.set_aspect('equal')

    # Draw gridlines
    for i in range(grid_size + 1):
        ax.axhline(i, color='gray', linewidth=1)
        ax.axvline(i, color='gray', linewidth=1) #gray

    # Hide the axes
    ax.axis('off')

    # Add text with state numbers
    if state_num:
        for state in range(env.n_states):
            # columns & rows
            y, x = state_to_xy(world, state)
            ax.text(x + 0.5, grid_size - y - 0.5, str(state), va='center', ha='center', color="black")

    # Coloring specific states
    if reward_color:
        
        for i in range(reward_info.shape[0]):
            state = reward_info[i][0]
            reward_value = reward_info[i][1]
            y, x = state_to_xy(world, state)
        
            if reward_value == 75:
                color = "#801515ff"
                alpha = 0.7
            elif reward_value == 50:
                color = "#d46a6aff"
                
                alpha = 0.7
            elif reward_value == 25:
                color = "#ffaaaaff" 
                alpha = 0.7
            else:
                color = "#ffffffff"
                alpha = 0.7

            ax.add_patch(plt.Rectangle((x, grid_size - y - 1), 1, 1, color=color, alpha=alpha))
            # Add color to reward state 87 for distribution 4
            ax.add_patch(plt.Rectangle((x, grid_size - y - 1), 1, 1, color=color, alpha=alpha))
            if reward_text:
                ax.text(x + 0.5, grid_size - y - 0.5, str(reward_value), va='center', ha='center', color="black")
    
    # Add boundaries
    if boundaries:
        add_boundaries(ax, world, boundaries, grid_size)

    # Add dashed lines for specific squares
    if dashed_squares is not None:
        
        for square in dashed_squares:
            y, x = state_to_xy(world, square)  
            ax.plot(
                [x, x+1, x+1, x, x],  # x-coordinates of the square boundary
                [grid_size - y - 1, grid_size - y - 1, grid_size - y, grid_size - y, grid_size - y - 1],  # y-coordinates
                linestyle='--', color='#D46A6A', linewidth=2
            )

    # Add thicker lines for the tiles
    ax.plot([0, 10], [5,5], 'gray', linewidth=3 )
    ax.plot([5, 5], [0,10], 'gray', linewidth=3 )
    # Add thicker lines for the edges
    ax.plot([0,10], [0,0], color='gray', linewidth=3)
    ax.plot([0,10], [10,10], color='gray', linewidth=3)
    ax.plot([0,0], [0,10], color='gray', linewidth=3)
    ax.plot([10,10], [0,10], color='gray', linewidth=3)


    # Add patch to the initial states
    if exp == 'baseline':
        ax.add_patch(plt.Rectangle((5, 5), 1, 1, color="#99d8c9"))
        ax.add_patch(plt.Rectangle((5, 4), 1, 1, color="#99d8c9"))
        ax.add_patch(plt.Rectangle((4, 5), 1, 1, color="#99d8c9"))
        ax.add_patch(plt.Rectangle((4, 4), 1, 1, color="#99d8c9"))
    elif exp == 'exp3':
        ax.add_patch(plt.Rectangle((3, 3), 1, 1, color="#99d8c9"))
        ax.add_patch(plt.Rectangle((4, 3), 1, 1, color="#99d8c9"))
        ax.add_patch(plt.Rectangle((5, 3), 1, 1, color="#99d8c9"))
        ax.add_patch(plt.Rectangle((6, 3), 1, 1, color="#99d8c9")) 
        ax.add_patch(plt.Rectangle((6, 4), 1, 1, color="#99d8c9"))
        ax.add_patch(plt.Rectangle((6, 5), 1, 1, color="#99d8c9"))
        ax.add_patch(plt.Rectangle((6, 6), 1, 1, color="#99d8c9")) 
        ax.add_patch(plt.Rectangle((5, 6), 1, 1, color="#99d8c9"))
        ax.add_patch(plt.Rectangle((4, 6), 1, 1, color="#99d8c9"))
        ax.add_patch(plt.Rectangle((3, 6), 1, 1, color="#99d8c9")) 
        ax.add_patch(plt.Rectangle((3, 5), 1, 1, color="#99d8c9"))
        ax.add_patch(plt.Rectangle((3, 4), 1, 1, color="#99d8c9"))

    # Add transition matrix arrows if provided
    # Draw the arrows for the transitions
    if transition_matrix is not None:
        directions = {"up": 0, "right": 1, "down": 2, "left": 3}
        #action_index = directions[direction]
        for d in directions.keys():
            action_index = directions[d]
            for state in range(env.n_states):
                # columns & rows
                y, x = state_to_xy(world, state)
                start_center = (x + 0.5, grid_size - y - 0.5)  # Adjusting origin to bottom left

                for end_state in range(env.n_states):
                    if transition_matrix[state, action_index, end_state] > 0:
                        end = state_to_xy(world, end_state)
                        color = "blue" if transition_matrix[state, action_index, end_state] == 1 else "magenta"
                        if state == end_state:
                            end_center = get_end_center(d, end, grid_size)
                            draw_arrow(ax, start_center, end_center, "red")
                        else: 
                            end_center = (end[1] + 0.5, grid_size - end[0] - 0.5)
                            draw_arrow(ax, start_center, end_center, color)

    if savefigpath:
        fig.savefig(f'saved/Figures/{savefigpath}.pdf', bbox_inches='tight')

def plot_figure1(env, world, state_num = None, boundaries = None, dashed_squares = None, transition_matrix = None, start_positions = 'baseline', ax = None):
    """
    Plots the world grid with optional attributes
    env: Environment object
    world: 2D numpy array representing the grid world
    state_num: Boolean to display state numbers
    boundaries: List of tuples with the boundaries
    dashed_squares: List of states to draw dashed squares
    transition_matrix: 3D numpy array with the transition matrix
    start_positions: String indicating the start locations ('baseline' for centered positions, 'exp3' for shifted start positions or none)
    savefigpath: Path to save the figure
    ax: Matplotlib axis object

    Returns:
    fig: Matplotlib figure object
    """
    
    grid_size = world.shape[0]
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
        #ax = plt.gca()  # Get the current axis if none is provided
    
    
    # Create a blank 8x8 white grid
    ax.set_xlim([0, grid_size])
    ax.set_ylim([0, grid_size])
    ax.set_aspect('equal')

    # Draw gridlines
    for i in range(grid_size + 1):
        ax.axhline(i, color='black', linewidth=1)
        ax.axvline(i, color='black', linewidth=1) 

    # Hide the axes
    ax.axis('off')

    # Add text with state numbers
    if state_num:
        for state in range(env.n_states):
            # columns & rows
            y, x = state_to_xy(world, state)
            ax.text(x + 0.5, grid_size - y - 0.5, str(state), va='center', ha='center', color="black")

    
    # Add boundaries
    if boundaries:
        add_boundaries(ax, world, boundaries, grid_size)

    # Add thicker lines for the edges
    ax.plot([0,10], [0,0], color='black', linewidth=5)
    ax.plot([0,10], [10,10], color='black', linewidth=5)
    ax.plot([0,0], [0,10], color='black', linewidth=5)
    ax.plot([10,10], [0,10], color='black', linewidth=5)

    for state in range(env.n_states):
            tile = env.get_tile_from_state(state)
            if tile == env.tileA:
                y, x = state_to_xy(world, state)
                ax.add_patch(plt.Rectangle((x, grid_size - y - 1), 1, 1, color="#f6cdc1"))#c49486")) #"#feffaa71"))
            elif tile == env.tileB:
                y, x = state_to_xy(world, state)
                ax.add_patch(plt.Rectangle((x, grid_size - y - 1), 1, 1, color="#B0A8C8")) #"#57285779"))
            elif tile == env.tileC:
                y, x = state_to_xy(world, state)
                ax.add_patch(plt.Rectangle((x, grid_size - y - 1), 1, 1, color="#e4a9a5")) #"#B87878")) #"#A2313179"))
            elif tile == env.tileD:
                y, x = state_to_xy(world, state)
                ax.add_patch(plt.Rectangle((x, grid_size - y - 1), 1, 1, color="#b8d4f0"))#b8d4f0"))

    # Add patch to the initial states
    if start_positions == 'baseline':
        ax.add_patch(plt.Rectangle((5, 5), 1, 1, color="#99d8c9"))
        ax.add_patch(plt.Rectangle((5, 4), 1, 1, color="#99d8c9"))
        ax.add_patch(plt.Rectangle((4, 5), 1, 1, color="#99d8c9"))
        ax.add_patch(plt.Rectangle((4, 4), 1, 1, color="#99d8c9"))
    elif start_positions == 'exp3':
        #ax.add_patch(plt.Rectangle((3, 3), 1, 1, color="#99d8c9"))
        ax.add_patch(plt.Rectangle((4, 3), 1, 1, color="#99d8c9"))
        ax.add_patch(plt.Rectangle((5, 3), 1, 1, color="#99d8c9"))
        ax.add_patch(plt.Rectangle((6, 3), 1, 1, color="#99d8c9")) 
        ax.add_patch(plt.Rectangle((6, 4), 1, 1, color="#99d8c9"))
        ax.add_patch(plt.Rectangle((6, 5), 1, 1, color="#99d8c9"))
        ax.add_patch(plt.Rectangle((6, 6), 1, 1, color="#99d8c9")) 
        ax.add_patch(plt.Rectangle((5, 6), 1, 1, color="#99d8c9"))
        ax.add_patch(plt.Rectangle((4, 6), 1, 1, color="#99d8c9"))
        ax.add_patch(plt.Rectangle((3, 6), 1, 1, color="#99d8c9")) 
        ax.add_patch(plt.Rectangle((3, 5), 1, 1, color="#99d8c9"))
        ax.add_patch(plt.Rectangle((3, 4), 1, 1, color="#99d8c9"))



    # Add dashed lines for specific squares
    if dashed_squares is not None:
        
        for square in dashed_squares:
            y, x = state_to_xy(world, square)  
            ax.plot(
                [x, x+1, x+1, x, x],  # x-coordinates of the square boundary
                [grid_size - y - 1, grid_size - y - 1, grid_size - y, grid_size - y, grid_size - y - 1],  # y-coordinates
                linestyle='--', color='#8b2929', linewidth=2
            )

    # Add transition matrix arrows if provided
    # Draw the arrows for the transitions
    if transition_matrix is not None:
        directions = {"up": 0, "right": 1, "down": 2, "left": 3}
        #action_index = directions[direction]
        for d in directions.keys():
            action_index = directions[d]
            for state in range(env.n_states):
                # columns & rows
                y, x = state_to_xy(world, state)
                start_center = (x + 0.5, grid_size - y - 0.5)  # Adjusting origin to bottom left

                for end_state in range(env.n_states):
                    if transition_matrix[state, action_index, end_state] > 0:
                        end = state_to_xy(world, end_state)
                        color = "blue" if transition_matrix[state, action_index, end_state] == 1 else "magenta"
                        if state == end_state:
                            end_center = get_end_center(d, end, grid_size)
                            draw_arrow(ax, start_center, end_center, "red")
                        else: 
                            end_center = (end[1] + 0.5, grid_size - end[0] - 0.5)
                            draw_arrow(ax, start_center, end_center, color)



    

def plot_individual_tiles(env, world, save_dir="saved/figures/world", start_positions=None, **kwargs):
    """
    Save the 4 quadrants (tiles) as separate images.
    
    Args:
        env, world: entire environment and world objects
        save_dir: folder to save the images
        **kwargs: All other arguments that your plot_figure1 function needs (e.g., reward_info, boundaries, etc.)
    """
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    grid_size = world.shape[0]   # Annahme: 10
    mid = grid_size / 2          # Annahme: 5

    # Coordinates of the 4 quadrants 
    quadrants = {
        "TopLeft":     (0, mid, mid, grid_size),
        "TopRight":    (mid, grid_size, mid, grid_size),
        "BottomLeft":  (0, mid, 0, mid),
        "BottomRight": (mid, grid_size, 0, mid)
    }

    for name, (x_min, x_max, y_min, y_max) in quadrants.items():
        fig, ax = plt.subplots(figsize=(4, 4))
        
        # Plot figure 1 on the entire world
        plot_figure1(env, world, ax=ax, start_positions=start_positions, **kwargs)
        
        # Crop to the respective quadrant by setting the limits of the axes
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        rect = plt.Rectangle(
            (x_min, y_min),   # Start (bottom left)
            mid, mid,         # width, height (here both 5)
            linewidth=7,      
            edgecolor='black',
            facecolor='none', 
            zorder=20,        
            clip_on=False     # Important: Prevents the line itself from being cropped
        )
        ax.add_patch(rect)
        
        ax.axis('off')
        
        # Save the figure
        fullpath = os.path.join(save_dir, f"tile_{name}.svg")
        plt.savefig(fullpath, bbox_inches='tight', pad_inches=0.0, dpi=300)
        plt.close(fig)



def plot_paths(env, world, reward_info, title, state_num = None, boundaries = None, reward_color = False, expert_path = None, agent_path = None, ax = None):

    grid_size = env.world_matrix.shape[0]

    #if ax is None:
        #ax = plt.gca()  # Get the current axis if none is provided
    fig, ax = plt.subplots()
    # Create a blank 8x8 white grid
    ax.set_xlim([0, grid_size])
    ax.set_ylim([0, grid_size])
    ax.set_aspect('equal')

    # Draw gridlines
    for i in range(grid_size + 1):
        ax.axhline(i, color='gray', linewidth=1)
        ax.axvline(i, color='gray', linewidth=1)

    # Hide the axes
    ax.axis('off')

    # Add text with state numbers
    if state_num:
        for state in range(env.n_states):
            # columns & rows
            y, x = state_to_xy(world, state)
            ax.text(x + 0.5, grid_size - y - 0.5, str(state), va='center', ha='center', color="black")

    # Coloring specific states
        # Coloring specific states
    for i in range(reward_info.shape[0]):
        
        state = reward_info[i][0]
        reward_value = reward_info[i][1]
        reward_sets = reward_info[i][2]
        y, x = state_to_xy(world, state)

        if reward_color:
            if reward_sets == 1:
                color = "#12169F"
                alpha = 1
            elif reward_sets == 2:
                color = "#1170B4"
                
                alpha = 1
            else:
                color = "#7E9ABF"
                reward_value = 25
                alpha = 1

            ax.add_patch(plt.Rectangle((x, grid_size - y - 1), 1, 1, color=color, alpha=alpha))
            

    # Add boundaries
    if boundaries:
        add_boundaries(ax, world, boundaries, grid_size)


    # Prepare a colormap and normalize
    original_cmap = plt.get_cmap('Reds')
    colors = original_cmap(np.linspace(0.1, 0.9, int(np.sum(~np.isnan(~np.isnan(agent_path))))))
    cmap = mcolors.LinearSegmentedColormap.from_list("darker_" + 'Reds', colors)
    #cmap = plt.get_cmap('Reds')  # You can change 'Blues' to any other colormap (e.g., 'Reds', 'Greens')
    norm = mcolors.Normalize(vmin=0, vmax=(0 + len(agent_path) - 1))

    # Draw the path
    agent_path = agent_path[~np.isnan(agent_path)]
    for i in range(len(agent_path) - 1):
        y0, x0 = state_to_xy( world, agent_path[i])
        y1, x1 = state_to_xy(world, agent_path[i + 1])
        color = colors[i]
        #color = cmap(norm(i))
        ax.plot([x0 + 0.5, x1 + 0.5], [grid_size - y0 - 0.5, grid_size - y1 - 0.5], color=color, linewidth=5)

    if expert_path is not None:
        # Prepare a colormap and normalize
        #original_cmap = plt.get_cmap('Blues')
        #colors = original_cmap(np.linspace(0.25, 1.0, 256))
        #cmap = mcolors.LinearSegmentedColormap.from_list("darker_" + 'Blues', colors)
        #cmap = plt.get_cmap('Reds')  # You can change 'Blues' to any other colormap (e.g., 'Reds', 'Greens')
        #norm = mcolors.Normalize(vmin=0, vmax=(0 + len(expert_path) - 1))
        expert_cmap = plt.get_cmap('Blues')
        expert_colors = expert_cmap(np.linspace(0.1, 0.9, int(np.sum(~np.isnan(expert_path)))))

        # Draw the path
        expert_path = expert_path[~np.isnan(expert_path)]
        for i in range(len(expert_path) - 1):
            y0, x0 = state_to_xy( world, expert_path[i])
            y1, x1 = state_to_xy(world, expert_path[i + 1])
            #color = cmap(norm(i))
            color = expert_colors[i]
            ax.plot([x0 + 0.5, x1 + 0.5], [grid_size - y0 - 0.5, grid_size - y1 - 0.5], color=color, linewidth=3 + i * 0.1)

    plt.title(title)


def load_data(file):
    # Load data from the expert
    with open(file, 'r') as json_file:
        dic = json.load(json_file)
    # Convert the saved lists to array
    for k in dic.keys():
        dic[k] = np.array(dic[k])
    return dic


def plot_performance(sum_reward, x_label, title, y_label, expert_gone):
    
    fig, ax = plt.subplots(figsize=(8, 5))

    #ax.plot(sum_reward.T, linestyle='-', alpha = 0.1, color="blue")
    ax.plot(np.arange(1, sum_reward.shape[1]+1), np.mean(sum_reward, axis = 0), color = "black", linewidth=2)
    #for i in range(sum_reward.shape[1]):
        #ax.plot(sum_reward[:,i])
    #ax.plot(np.mean(sum_reward, axis = 1), label = "mean")
    
    ax.set_xticks(np.insert(np.arange(5, sum_reward.shape[1]+1, 5), 0, 1))
    if expert_gone != None:
        ax.axvline(expert_gone,linestyle='--' )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    #ax.spines[['right', 'top']].set_visible(False)
    #plt.title(title)
    #plt.legend()

    #plt.show()
    # plt.savefig('saved_plot.png', bbox_inches='tight')
    # plt.close()
    return fig, ax


    
def plot_value(value):
    cmap = matplotlib.cm.jet
    cmap.set_bad('purple',1.)

    fig, ax = plt.subplots(1,1, figsize=(15, 3))

    #im = plt.imshow(value.T, interpolation='nearest', cmap=cmap)
    #plt.colorbar(im)
    sns.heatmap(value.T, ax=ax)
    plt.title("Value Function")

# plot performance
def plot_with_se(ax, x, data, color, label, linestyle='-'):
        mean = np.mean(data, axis=0)
        se = np.std(data, axis=0) / np.sqrt(data.shape[0])  # Standard error
        ax.plot(x, mean, color=color, linewidth=2, linestyle=linestyle, label=label)
        ax.fill_between(x, mean - se, mean + se, color=color, alpha=0.2)  # Shaded region

# ALL MODELS STEPS
def all_models(mbased_metric, mfree_metric, n_episodes, training, exp, ylabel, expert):
    
    """
    Plot the performance of all models with SEs
    mbased_metric: List of arrays with the metrics for model based
    mfree_metric: List of arrays with the metrics for model free
    n_episodes: Number of episodes
    training: Percentage of training
    exp: String indicating the experiment type (e.g., 'baseline', 'exp2', 'exp3') for legend placements
    ylabel: Label of the y axis
    expert: Expert data

    Returns:
    fig: Matplotlib figure object
    """
    
    expert_gone = int(n_episodes * training)
    fig, ax = plt.subplots(figsize=(6, 4))

    # Expert performance line
    ax.axhline(
    expert,
    color="black",
    linestyle="-",
    linewidth=2,
    xmin=0,
    xmax=19
    )


    assert mbased_metric[0].shape[1] == mfree_metric[0].shape[1], "The number of episodes must be the same for all models"
    xs = np.arange(1, mbased_metric[0].shape[1]+1)

    plot_with_se(ax, xs, mbased_metric[0], "dimgray", "Asocial Learning")
    plot_with_se(ax, xs, mbased_metric[1], "darkorange", "Decision Bias")
    plot_with_se(ax, xs, mbased_metric[2], "hotpink", "Value Shaping")

    plot_with_se(ax, xs, mfree_metric[0], "dimgray", "Asocial Learning", linestyle='--')
    plot_with_se(ax, xs, mfree_metric[1], "darkorange", "Decision Bias", linestyle='--')
    plot_with_se(ax, xs, mfree_metric[2], "hotpink", "Value Shaping", linestyle='--')

    # Optional label
    ax.text(
        n_episodes * 0.9,  # near left edge
        expert*0.85,
        "Expert",
        ha="left",
        va="bottom",
        fontsize=14,
        color="black"
    )
    ax.set_xlabel("Episodes", fontsize = 20)
    ax.set_ylabel(ylabel, fontsize = 20)
    ax.set_xticks(np.insert(np.arange(5, n_episodes+1, 5), 0, 1))
    ax.set_yticks(ax.get_yticks())
    ax.set_ylim(-45, 70)
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)

    # Separator line for training and testing phase
    ax.axvline(expert_gone, linestyle = '--' , color = "black")
    ax.text(expert_gone / 2, ax.get_ylim()[1] * 0.95,
       "Training", ha='center', va='bottom', fontsize=20, color = "black")

    ax.text((expert_gone + n_episodes) / 2, ax.get_ylim()[1] * 0.95,
   "Test", ha='center', va='bottom', fontsize=20, color = "black")


    # Setting the default font size for xticks
    plt.rcParams['xtick.labelsize'] = 15

    # Custom legend
    type_of_model = [
        Line2D([0], [0], color = "black", linewidth=1.5, label = "MB" ),
        Line2D([0], [0], color = "black", linewidth=1.5, linestyle='--' , label = "MF"),
        ]


    social_strategy = [
    Line2D([0], [0], color = "dimgray", linewidth=1.5, label = "AS"),
    Line2D([0], [0], color = "darkorange", linewidth=1.5, label = "DB"),
    Line2D([0], [0], color = "hotpink", linewidth=1.5, label = "VS"),
        ]
    dummy = Line2D([], [], linestyle='None', label='')


    # Baseline legend
    if exp == 'baseline':
        bold_title = FontProperties(weight='bold', size=14)
        legend1 = ax.legend(handles=social_strategy, title = "SL", loc='upper left', bbox_to_anchor=(0.51, 0.75), fontsize = 12, title_fontproperties=bold_title,frameon=True)
        ax.add_artist(legend1)  # Add the first legend to the axes
        legend2 = ax.legend(handles=type_of_model + [dummy], title = "RL", loc='upper left', bbox_to_anchor=(0.75, 0.75), fontsize = 12, title_fontproperties=bold_title, frameon=True)
        ax.text(0.74, 0.55, "×", transform=ax.transAxes, fontsize=18, ha= 'center', va = 'center')
    if exp == 'exp2':
        bold_title = FontProperties(weight='bold', size=14)
        legend1 = ax.legend(handles=social_strategy, title = "SL", loc='upper left', bbox_to_anchor=(0.51, 0.65), fontsize = 12, title_fontproperties=bold_title,frameon=True)
        ax.add_artist(legend1)  # Add the first legend to the axes
        legend2 = ax.legend(handles=type_of_model, title = "RL", loc='upper left', bbox_to_anchor=(0.75, 0.65), fontsize = 12, title_fontproperties=bold_title, frameon=True)
        ax.text(0.74, 0.45, "×", transform=ax.transAxes, fontsize=18, ha= 'center', va = 'center')


    # Setting the default font size for xticks
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return fig


# To plot for time steps
def visualize_value_updates_with_world(world, value_list, timesteps=40):
    """
    Visualize value updates as heatmaps on a grid-based world, considering unordered states.

    Args:
        world (np.ndarray): 2D grid specifying the layout of the world.
        value_list (list of np.ndarray): List of value arrays (100x64) for each timestep.
        state_to_xy (function): Function mapping state numbers to (x, y) grid coordinates.
        timesteps (int): Number of timesteps to visualize.
    """
    grid_size = world.shape[0]
    num_columns = 4
    num_rows = int(np.ceil(timesteps / num_columns))  # Calculate required number of rows

    # Validate inputs
    num_states = np.prod(world.shape)
    #assert len(value_list) >= timesteps, "Value list must have at least `timesteps` entries."
    #assert value_list[0].shape == (num_states, 64), "Value arrays must be 100x64."

    # Create subplots
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(16, 4 * num_rows))
    axes = axes.flatten()  # Flatten axes for easier iteration

    for t in range(timesteps):
        # Get the value array for the current timestep
        value_matrix = value_list[t] # value for each of the 100 states
        
        # Create a heatmap array matching the world grid size
        value_heatmap = np.ones_like(world, dtype=float)

        # Populate heatmap values and text
        for state in range(num_states):
            y, x = state_to_xy(world, state)
            if np.all(value_matrix[state] == 1):  # If all action values are 1
                value_heatmap[y, x] = 1
            else:  # Take the maximum value different from 1
                max_value = np.max(value_matrix[state][value_matrix[state] != 1])
                value_heatmap[y, x] = max_value

        # Plot the heatmap for the current timestep
        ax = axes[t]
        ax.set_xlim([0, grid_size])
        ax.set_ylim([0, grid_size])
        ax.set_aspect("equal")

        # Draw gridlines
        for i in range(grid_size + 1):
            ax.axhline(i, color="gray", linewidth=1)
            ax.axvline(i, color="gray", linewidth=1)

        # Hide axes
        ax.axis("off")

        # Add text with state numbers
        for state in range(num_states):
            y, x = state_to_xy(world, state)
            ax.text(
                x + 0.5,
                grid_size - y - 0.5,
                str(state),
                va="center",
                ha="center",
                color="black",
            )

        # Display the heatmap overlay
        heatmap = ax.imshow(
            value_heatmap,
            cmap="viridis",
            extent=(0, grid_size, 0, grid_size),
            origin="upper",
            alpha=0.6,
        )

         # Add individual color bars next to each heatmap
        cbar = fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Value")

        ax.set_title(f"Timestep {t}") #{t+1}
        


    # Hide unused subplots
    for t in range(timesteps, len(axes)):
        axes[t].axis("off")

    # Adjust layout
    fig.tight_layout()

    # Add a single color bar for the entire figure
    #cbar_ax = fig.add_axes([1, 0.15, 0.02, 0.7])  # Adjust position to the right
    #fig.colorbar(heatmap, cax=cbar_ax, label="Value")

    plt.show()



def save_legend_only(save_path):
    # Legend elements
    type_of_model = [
        Line2D([0], [0], color="black", linewidth=1.5, label="Model-based"),
        Line2D([0], [0], color="black", linewidth=1.5, linestyle="--", label="Model-free"),
    ]

    social_strategy = [
        Line2D([0], [0], color="dimgray", linewidth=1.5, label="Asocial learning"),
        Line2D([0], [0], color="darkorange", linewidth=1.5, label="Decision bias"),
        Line2D([0], [0], color="hotpink", linewidth=1.5, label="Value shaping"),
    ]

    handles = type_of_model + social_strategy

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.axis("off")

    legend1 = ax.legend(
        handles=social_strategy,
        loc="upper left",
        bbox_to_anchor=(0, 1),
        fontsize=15,
        frameon=False
    )
    ax.add_artist(legend1)

    ax.legend(
        handles=type_of_model,
        loc="upper left",
        bbox_to_anchor=(0, 0.55),
        fontsize=15,
        frameon=False
    )

    # Save legend only
    fig.savefig(
        save_path,
        bbox_inches="tight",
        pad_inches=0
    )
    plt.close(fig)

