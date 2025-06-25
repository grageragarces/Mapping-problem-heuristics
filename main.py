import hdh 
from hdh.converters.convert_from_qasm import from_qasm
import networkx as nx
import numpy as np
from collections import defaultdict
import pymetis

# --- 1. Circuit Graph Generation ---
def create_circuit_representation_from_qasm(qasm_file_path):
    try:
        circuit_hdh: hdh = from_qasm('file', qasm_file_path)
    except FileNotFoundError:
        print(f"Error: QASM file not found at {qasm_file_path}")
        return None, None, None
    except Exception as e:
        print(f"Error loading QASM file with hdh: {e}")
        return None, None, None

    initial_qubit_groups: dict[int, list[str]] = defaultdict(list)
    circuit_dependencies_for_cost: list[tuple[str, str]] = []
    last_op_on_qubit: dict[int, str] = {}

    op_hdh_nodes_with_time: list[tuple[str, int]] = []
    for hdh_node_id in circuit_hdh.S:
        op_hdh_nodes_with_time.append((hdh_node_id, circuit_hdh.time_map.get(hdh_node_id, 0)))
    op_hdh_nodes_with_time.sort(key=lambda x: x[1])

    for op_hdh_node_id, _ in op_hdh_nodes_with_time:
        state_nodes_involved: list[str] = []
        qubits_involved_in_op = set()

        for h_edge_frozenset in circuit_hdh.C:
            if op_hdh_node_id in h_edge_frozenset and circuit_hdh.tau.get(h_edge_frozenset) == 'q':
                for member_hdh_node_id in h_edge_frozenset:
                    if member_hdh_node_id != op_hdh_node_id and circuit_hdh.sigma.get(member_hdh_node_id) == 'q':
                        state_nodes_involved.append(member_hdh_node_id)
                        try:
                            if '_t' in member_hdh_node_id:
                                q_idx_str = member_hdh_node_id[1:].split('_t')[0]
                            else:
                                q_idx_str = member_hdh_node_id.split('_')[1]
                            q_idx = int(q_idx_str)
                            qubits_involved_in_op.add(q_idx)
                        except (ValueError, IndexError):
                            continue

        for i in range(len(state_nodes_involved)):
            for j in range(i + 1, len(state_nodes_involved)):
                n1 = state_nodes_involved[i]
                n2 = state_nodes_involved[j]
                circuit_dependencies_for_cost.append((n1, n2))

        for qid in qubits_involved_in_op:
            if qid in last_op_on_qubit:
                circuit_dependencies_for_cost.append((last_op_on_qubit[qid], op_hdh_node_id))
            last_op_on_qubit[qid] = op_hdh_node_id
            initial_qubit_groups[qid].append(op_hdh_node_id)

    return circuit_hdh, initial_qubit_groups, circuit_dependencies_for_cost 


# --- 2. Network Graph Creation ---

def get_operation_nodes(hdh_obj):
    op_nodes = []
    for node_id in hdh_obj.S:
        if hdh_obj.sigma[node_id] == 'q' or hdh_obj.sigma[node_id] == 'c':
            continue
        for h in hdh_obj.C:
            if node_id in h and hdh_obj.tau[h] == 'q':
                op_nodes.append(node_id)
                break
    return op_nodes


def create_linear_network_graph(num_devices, capacities):
    network_graph = nx.Graph()

    if len(capacities) != num_devices:
        raise ValueError("Number of capacities must match the number of devices.")

    for i in range(num_devices):
        node_id = f"chip_{i}"
        network_graph.add_node(node_id, capacity=capacities[i], current_load=0)
        network_graph.add_edge(node_id, node_id) 

    for i in range(num_devices - 1):
        network_graph.add_edge(f"chip_{i}", f"chip_{i+1}")

    return network_graph

# --- 3. Cost Calculation Helper ---
def precompute_all_pairs_shortest_paths(graph):
    return dict(nx.all_pairs_shortest_path_length(graph))

# --- 4. Initial Mapping Strategy ---
def extract_qubit_index(node_id: str) -> int:
    try:
        return int(node_id[1:].split('_')[0])
    except Exception:
        return -1

def create_initial_mapping(circuit_hdh_obj: hdh, network_graph: nx.Graph, initial_qubit_groups: dict[int, list[str]]):
    mapping: dict[str, str] = {}
    network_node_loads: dict[str, int] = {node: 0 for node in network_graph.nodes()}
    network_nodes = list(network_graph.nodes())
    np.random.shuffle(network_nodes)

    qubit_to_nodes = defaultdict(list)
    for node_id in circuit_hdh_obj.S:
        if circuit_hdh_obj.sigma.get(node_id) == 'q':
            qid = extract_qubit_index(node_id)
            if qid >= 0:
                qubit_to_nodes[qid].append(node_id)

    for qid, node_ids in qubit_to_nodes.items():
        for chip_id in network_nodes:
            if network_node_loads[chip_id] < network_graph.nodes[chip_id]['capacity']:
                for nid in node_ids:
                    mapping[nid] = chip_id
                network_node_loads[chip_id] += 1
                break

    for node_id in circuit_hdh_obj.S:
        if circuit_hdh_obj.sigma.get(node_id) in ['q', 'c']:
            continue

        connected_chips = []
        for hedge in circuit_hdh_obj.C:
            if node_id in hedge and circuit_hdh_obj.tau.get(hedge) == 'q':
                for neighbor in hedge:
                    if neighbor != node_id and circuit_hdh_obj.sigma.get(neighbor) == 'q':
                        if neighbor in mapping:
                            connected_chips.append(mapping[neighbor])
        if connected_chips:
            preferred_chip = max(set(connected_chips), key=connected_chips.count)
            mapping[node_id] = preferred_chip
        else:
            mapping[node_id] = np.random.choice(network_nodes)

    return mapping, network_node_loads

# --- 5. Local Search Implementation ---

def calculate_total_cost(mapping: dict[str, str], circuit_dependencies_for_cost: list[tuple[str, str]], shortest_paths_matrix: dict[str, dict[str, int]]):
    total_cost = 0
    for u_c_op, v_c_op in circuit_dependencies_for_cost:
        mapped_u = mapping.get(u_c_op)
        mapped_v = mapping.get(v_c_op)

        if mapped_u is None or mapped_v is None:
            continue
        if mapped_u == mapped_v:
            continue

        try:
            cost_for_dependency = shortest_paths_matrix[mapped_u][mapped_v]
        except KeyError:
            cost_for_dependency = float('inf') 

        total_cost += cost_for_dependency
    return total_cost

def run_local_search(circuit_hdh_obj: hdh, network_graph: nx.Graph, initial_mapping: dict[str, str], network_node_loads: dict[str, int], shortest_paths_matrix: dict[str, dict[str, int]], circuit_dependencies_for_cost: list[tuple[str, str]], max_iterations: int = 10000, patience: int = 5):
    current_mapping = initial_mapping.copy()
    current_network_loads = network_node_loads.copy() 
    current_cost = calculate_total_cost(current_mapping, circuit_dependencies_for_cost, shortest_paths_matrix)

    #print(f"\nStarting Local Search. Initial cost: {current_cost}")

    iteration = 0
    no_improvement_rounds = 0

    while iteration < max_iterations and no_improvement_rounds < patience:
        iteration += 1
        move_made = False

        state_nodes = [node_id for node_id in circuit_hdh_obj.S if circuit_hdh_obj.sigma.get(node_id) == 'q']
        np.random.shuffle(state_nodes)

        #print(f"[Iteration {iteration}] Considering {len(state_nodes)} state nodes.")

        for state_id in state_nodes:
            if state_id not in current_mapping:
                continue

            current_chip = current_mapping[state_id]
            neighbors = list(network_graph.neighbors(current_chip))
            np.random.shuffle(neighbors)

            for target_chip in neighbors:
                if target_chip == current_chip:
                    continue

                if current_network_loads[target_chip] >= network_graph.nodes[target_chip]['capacity']:
                    continue

                temp_mapping = current_mapping.copy()
                temp_mapping[state_id] = target_chip

                new_cost = calculate_total_cost(temp_mapping, circuit_dependencies_for_cost, shortest_paths_matrix)

                if new_cost < current_cost:
                    current_mapping = temp_mapping
                    current_network_loads[current_chip] -= 1
                    current_network_loads[target_chip] += 1
                    current_cost = new_cost
                    move_made = True
                    #print(f"\n Move {state_id}: {current_chip} -> {target_chip} | Cost: {new_cost}")
                    break  # restart outer loop
            if move_made:
                break

        if not move_made:
            no_improvement_rounds += 1
        else:
            no_improvement_rounds = 0

    # print(f"\nLocal Search finished after {iteration} iterations. Final cost: {current_cost}")
    # print("Final Mapping:")
    # for c_node, n_node in current_mapping.items():
    #     print(f"  {c_node} -> {n_node}")
    print(f"\nNumber of iterations: {iteration}")
    # print("Final Network Node Loads:")
    # for n_node, load in current_network_loads.items():
    #     print(f"  {n_node}: {load}/{network_graph.nodes[n_node]['capacity']}")
    
    return current_mapping, current_cost, current_network_loads

# --- 6. Partitioning with METIS (opposition) ---
def partition_hdh_with_metis(circuit_hdh_obj, num_devices, circuit_dependencies, shortest_paths_matrix):

    # Step 1: Extract state nodes
    state_nodes = sorted([n for n in circuit_hdh_obj.S if circuit_hdh_obj.sigma[n] == 'q'])
    node_idx_map = {node_id: idx for idx, node_id in enumerate(state_nodes)}

    # Step 2: Build adjacency list from hyperedges
    adjacency = {i: set() for i in range(len(state_nodes))}
    for hedge in circuit_hdh_obj.C:
        if circuit_hdh_obj.tau[hedge] != 'q':
            continue
        involved = [n for n in hedge if n in node_idx_map]
        for i in range(len(involved)):
            for j in range(i + 1, len(involved)):
                idx_i = node_idx_map[involved[i]]
                idx_j = node_idx_map[involved[j]]
                adjacency[idx_i].add(idx_j)
                adjacency[idx_j].add(idx_i)

    metis_graph = [list(adjacency[i]) for i in range(len(state_nodes))]

    # Step 3: Run METIS
    _, partitions = pymetis.part_graph(num_devices, adjacency=metis_graph)

    # Step 4: Build mapping and compute cost
    metis_mapping = {
        state_nodes[i]: f"chip_{partitions[i]}" for i in range(len(state_nodes))
    }

    # Step 5: Compute cost
    cost = calculate_total_cost(metis_mapping, circuit_dependencies, shortest_paths_matrix)
    return cost
    
# --- Main Execution ---
if __name__ == "__main__":
    qasm_file = "ghz_indep_qiskit_128.qasm" #TODO: replace this with your QASM file path

    # 1. Circuit Representation 
    # returns the hdh object itself, qubit groups, and a list of dependencies.
    circuit_hdh, initial_qubit_groups, circuit_dependencies = create_circuit_representation_from_qasm(qasm_file)
    if circuit_hdh is None:
        exit() 

    # 2. Network Graph 
    num_devices = 4
    
    # calculates total circuit operation nodes from hdh for capacity check
    total_logical_qubits = len(initial_qubit_groups)
    per_chip_capacity = int(np.ceil(total_logical_qubits / num_devices)*1.2) # 20% extra capacity 
    capacities_per_chip = [per_chip_capacity] * num_devices

    if sum(capacities_per_chip) < total_logical_qubits:
        # print(f"WARNING: Total network capacity ({sum(capacities_per_chip)}) is less than total logical qubits ({total_logical_qubits}). The problem might be infeasible.")
        # print(f"Adjusting capacities for demonstration: Each chip will have capacity ceil(total_qubits / num_devices).")
        capacities_per_chip = [int(np.ceil(total_logical_qubits / num_devices))] * num_devices
        # print(f"Adjusted capacities to ensure feasibility: {capacities_per_chip}")

    network_graph = create_linear_network_graph(num_devices, capacities_per_chip)

    # 3. Precompute shortest paths for the network graph 
    shortest_paths_matrix = precompute_all_pairs_shortest_paths(network_graph)
    # print("\nNetwork Shortest Paths:")
    # for s, paths in shortest_paths_matrix.items():
    #     for t, dist in paths.items():
            # print(f"  Dist({s}, {t}) = {dist}")

    # 4. Initial Mapping 
    initial_mapping, network_node_loads = create_initial_mapping(circuit_hdh, network_graph, initial_qubit_groups)

    # 5. Run Local Search 
    final_mapping, final_cost, final_network_loads = run_local_search(
        circuit_hdh,
        network_graph,
        initial_mapping,
        network_node_loads,
        shortest_paths_matrix,
        circuit_dependencies 
    )

    # print(f"\n Final communication cost: {final_cost}")

    # Sanity check: cost should never exceed number of dependencies × max network distance
    max_possible_cost = len(circuit_dependencies) * max(
        max(paths.values()) for paths in shortest_paths_matrix.values()
    )
    if final_cost > max_possible_cost:
        print(f"WARNING: Final cost ({final_cost}) exceeds max possible cost estimate ({max_possible_cost})")
    else:
        print(f"Sanity check passed: Final cost ({final_cost}) ≤ max possible cost estimate ({max_possible_cost}).")

    cross_chip_dependencies = [
        (u, v) for (u, v) in circuit_dependencies
        if final_mapping.get(u) != final_mapping.get(v)
    ]
    if final_cost > len(cross_chip_dependencies):
        print(f"NOTE: Final cost ({final_cost}) exceeds number of cross-chip dependencies ({len(cross_chip_dependencies)}).")
    else:
        print(f"Sanity check passed: Final cost ({final_cost}) ≤ cross-chip dependencies ({len(cross_chip_dependencies)}).")

    opposing = partition_hdh_with_metis(circuit_hdh,
        num_devices,
        circuit_dependencies,
        shortest_paths_matrix
    )
    
    print(f"\nOpposing partitioning result: {opposing}")
    if final_cost < opposing:
        print(f"Final cost ({final_cost}) is less than opposing partitioning cost ({opposing}).")
    else:
        print(f"Final cost ({final_cost}) is greater than or equal to opposing partitioning cost ({opposing}).")