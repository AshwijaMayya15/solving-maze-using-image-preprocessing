import cv2
import numpy as np 
import queue as Queue
# Load the image
image_path = './20cells.png' # Replace with the actual path to your image
image = cv2.imread(image_path)


# Check if the image was successfully loaded
if image is None:
    print("Failed to load the image.")
else:
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding to convert the image to binary
    _, threshold = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Apply Canny edge detection
    edges = cv2.Canny(threshold, 50, 150)

    # Maze representation using a graph data structure
    maze_height, maze_width = edges.shape
    maze_graph = np.zeros((maze_height, maze_width), dtype=int)

    # Iterate through the maze edges
    for y in range(maze_height):
        for x in range(maze_width):
            if edges[y, x] > 0:  # Edge detected
                maze_graph[y, x] = 1  # Set the corresponding cell as a node

    # Display the original image, preprocessed image, and edges
    cv2.imshow("Original Image", image)
    cv2.imshow("Preprocessed Image", threshold)
    cv2.imshow("Maze Edges", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Pathfinding algorithm (modified BFS)
    start_node = (0, 0)  # Define the start node
    goal_node = (maze_height - 1, maze_width - 1)  # Define the goal node

    visited = np.zeros((maze_height, maze_width), dtype=int)
    parent = np.zeros((maze_height, maze_width, 2), dtype=int)
    parent[start_node] = (-1, -1)

    q = Queue()
    q.put(start_node)

    while not q.empty():
        current_node = q.get()

        if current_node == goal_node:
            break

        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            next_node = (current_node[0] + dy, current_node[1] + dx)

            if 0 <= next_node[0] < maze_height and 0 <= next_node[1] < maze_width:
                if maze_graph[next_node] == 1 and visited[next_node] == 0:
                    visited[next_node] = 1
                    parent[next_node] = current_node
                    q.put(next_node)

    # Trace back the path from goal to start
    path = []
    current_node = goal_node
    while current_node != (-1, -1):
        path.append(current_node)
        current_node = parent[current_node]

    # Print the path
    if len(path) > 0:
        path.reverse()
        for node in path:
            print(node)
    else:
        print("No path found")