"""C ABI kernels for mojo-graph-tool.

Graphs cross the boundary as CSR arrays.  The Python layer owns both graph and
scratch storage; these kernels allocate nothing and therefore have no lifetime
rules beyond the call.
"""

from std.sys import simd_width_of

comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]


@export("mgt_pagerank")
def mgt_pagerank(
    offsets_addr: Int,
    incoming_addr: Int,
    weight_addr: Int,
    out_strength_addr: Int,
    pers_addr: Int,
    rank_addr: Int,
    scratch_addr: Int,
    contribution_addr: Int,
    n: Int,
    weighted: Int,
    damping: Float64,
    epsilon: Float64,
    max_iter: Int,
) abi("C") -> Int:
    var offsets = IPtr(unsafe_from_address=offsets_addr)
    var incoming = IPtr(unsafe_from_address=incoming_addr)
    var weight = FPtr(unsafe_from_address=weight_addr)
    var out_strength = FPtr(unsafe_from_address=out_strength_addr)
    var pers = FPtr(unsafe_from_address=pers_addr)
    var rank = FPtr(unsafe_from_address=rank_addr)
    var scratch = FPtr(unsafe_from_address=scratch_addr)
    var contribution = FPtr(unsafe_from_address=contribution_addr)
    var iterations = max_iter if max_iter > 0 else 1000
    comptime W = simd_width_of[DType.float64]()
    var vector_end = n - n % W
    for step in range(iterations):
        var dangling = 0.0
        for source in range(0, vector_end, W):
            var strengths = out_strength.load[width=W](source)
            var ranks = rank.load[width=W](source)
            var dangling_mask = strengths.eq(0.0)
            dangling += dangling_mask.select(ranks, 0.0).reduce_add()
            var denominators = dangling_mask.select(1.0, strengths)
            contribution.store(
                source,
                dangling_mask.select(0.0, damping * ranks / denominators),
            )
        for source in range(vector_end, n):
            var strength = out_strength[source]
            if strength == 0.0:
                dangling += rank[source]
                contribution[source] = 0.0
            else:
                contribution[source] = damping * rank[source] / strength

        @parameter
        def update_weighted_vertex(vertex: Int):
            var value = (1.0 - damping) * pers[vertex] + damping * dangling * pers[vertex]
            for slot in range(Int(offsets[vertex]), Int(offsets[vertex + 1])):
                var source = Int(incoming[slot])
                value += contribution[source] * weight[slot]
            scratch[vertex] = value

        @parameter
        def update_unweighted_vertex(vertex: Int):
            var value = (1.0 - damping) * pers[vertex] + damping * dangling * pers[vertex]
            for slot in range(Int(offsets[vertex]), Int(offsets[vertex + 1])):
                value += contribution[Int(incoming[slot])]
            scratch[vertex] = value

        if weighted:
            for vertex in range(n):
                update_weighted_vertex(vertex)
        else:
            for vertex in range(n):
                update_unweighted_vertex(vertex)

        var error = 0.0
        for vertex in range(0, vector_end, W):
            var values = scratch.load[width=W](vertex)
            error += abs(values - rank.load[width=W](vertex)).reduce_add()
            rank.store(vertex, values)
        for vertex in range(vector_end, n):
            var value = scratch[vertex]
            error += abs(value - rank[vertex])
            rank[vertex] = value
        if error < epsilon:
            return step + 1
    return iterations


@export("mgt_bfs")
def mgt_bfs(
    offsets_addr: Int, neighbors_addr: Int, distance_addr: Int, queue_addr: Int,
    n: Int, source: Int, max_dist: Int,
) abi("C") -> Int:
    var offsets = IPtr(unsafe_from_address=offsets_addr)
    var neighbors = IPtr(unsafe_from_address=neighbors_addr)
    var distance = IPtr(unsafe_from_address=distance_addr)
    var queue = IPtr(unsafe_from_address=queue_addr)
    for vertex in range(n):
        distance[vertex] = -1
    if source < 0 or source >= n:
        return 0
    var head = 0
    var tail = 1
    queue[0] = Int64(source)
    distance[source] = 0
    while head < tail:
        var vertex = Int(queue[head])
        head += 1
        var level = Int(distance[vertex])
        if max_dist >= 0 and level >= max_dist:
            continue
        for slot in range(Int(offsets[vertex]), Int(offsets[vertex + 1])):
            var neighbor = Int(neighbors[slot])
            if distance[neighbor] == -1:
                distance[neighbor] = Int64(level + 1)
                queue[tail] = Int64(neighbor)
                tail += 1
    return tail


@export("mgt_components")
def mgt_components(
    offsets_addr: Int, neighbors_addr: Int, labels_addr: Int, queue_addr: Int, n: Int,
) abi("C") -> Int:
    var offsets = IPtr(unsafe_from_address=offsets_addr)
    var neighbors = IPtr(unsafe_from_address=neighbors_addr)
    var labels = IPtr(unsafe_from_address=labels_addr)
    var queue = IPtr(unsafe_from_address=queue_addr)
    for vertex in range(n):
        labels[vertex] = -1
    var count = 0
    for root in range(n):
        if labels[root] != -1:
            continue
        var head = 0
        var tail = 1
        queue[0] = Int64(root)
        labels[root] = Int64(count)
        while head < tail:
            var vertex = Int(queue[head])
            head += 1
            for slot in range(Int(offsets[vertex]), Int(offsets[vertex + 1])):
                var neighbor = Int(neighbors[slot])
                if labels[neighbor] == -1:
                    labels[neighbor] = Int64(count)
                    queue[tail] = Int64(neighbor)
                    tail += 1
        count += 1
    return count


@export("mgt_kcore")
def mgt_kcore(
    offsets_addr: Int, neighbors_addr: Int, degree_addr: Int, core_addr: Int,
    bin_addr: Int, order_addr: Int, position_addr: Int, n: Int,
) abi("C"):
    var offsets = IPtr(unsafe_from_address=offsets_addr)
    var neighbors = IPtr(unsafe_from_address=neighbors_addr)
    var degree = IPtr(unsafe_from_address=degree_addr)
    var core = IPtr(unsafe_from_address=core_addr)
    var bins = IPtr(unsafe_from_address=bin_addr)
    var order = IPtr(unsafe_from_address=order_addr)
    var position = IPtr(unsafe_from_address=position_addr)
    var largest = 0
    for vertex in range(n):
        degree[vertex] = offsets[vertex + 1] - offsets[vertex]
        if Int(degree[vertex]) > largest:
            largest = Int(degree[vertex])
    for degree_value in range(largest + 1):
        bins[degree_value] = 0
    for vertex in range(n):
        bins[Int(degree[vertex])] += 1
    var start = 0
    for degree_value in range(largest + 1):
        var count = Int(bins[degree_value])
        bins[degree_value] = Int64(start)
        start += count
    for vertex in range(n):
        var degree_value = Int(degree[vertex])
        position[vertex] = bins[degree_value]
        order[Int(position[vertex])] = Int64(vertex)
        bins[degree_value] += 1
    var previous = 0
    for degree_value in range(largest + 1):
        var next = Int(bins[degree_value])
        bins[degree_value] = Int64(previous)
        previous = next
    for index in range(n):
        var vertex = Int(order[index])
        var vertex_degree = Int(degree[vertex])
        core[vertex] = Int64(vertex_degree)
        for slot in range(Int(offsets[vertex]), Int(offsets[vertex + 1])):
            var neighbor = Int(neighbors[slot])
            if Int(degree[neighbor]) > vertex_degree:
                var neighbor_degree = Int(degree[neighbor])
                var neighbor_position = Int(position[neighbor])
                var first_position = Int(bins[neighbor_degree])
                var first_vertex = Int(order[first_position])
                if neighbor != first_vertex:
                    order[neighbor_position] = Int64(first_vertex)
                    order[first_position] = Int64(neighbor)
                    position[neighbor] = Int64(first_position)
                    position[first_vertex] = Int64(neighbor_position)
                bins[neighbor_degree] += 1
                degree[neighbor] -= 1


def count_clustering_range(
    offsets: IPtr,
    neighbors: IPtr,
    triangle: IPtr,
    local: FPtr,
    first: Int,
    last: Int,
):
    for vertex in range(first, last):
        var start = Int(offsets[vertex])
        var stop = Int(offsets[vertex + 1])
        var degree = stop - start
        if degree < 2:
            triangle[vertex] = 0
            local[vertex] = 0.0
            return
        var count = 0
        for edge_slot in range(start, stop - 1):
            var neighbor = Int(neighbors[edge_slot])
            var left = edge_slot + 1
            var right = Int(offsets[neighbor])
            var right_stop = Int(offsets[neighbor + 1])
            while left < stop and right < right_stop:
                var left_neighbor = Int(neighbors[left])
                var right_neighbor = Int(neighbors[right])
                if left_neighbor < right_neighbor:
                    left += 1
                elif left_neighbor > right_neighbor:
                    right += 1
                else:
                    count += 1
                    left += 1
                    right += 1
        triangle[vertex] = Int64(count)
        local[vertex] = Float64(2 * count) / Float64(degree * (degree - 1))


@export("mgt_local_clustering_range")
def mgt_local_clustering_range(
    offsets_addr: Int,
    neighbors_addr: Int,
    triangle_addr: Int,
    local_addr: Int,
    first: Int,
    last: Int,
) abi("C"):
    count_clustering_range(
        IPtr(unsafe_from_address=offsets_addr),
        IPtr(unsafe_from_address=neighbors_addr),
        IPtr(unsafe_from_address=triangle_addr),
        FPtr(unsafe_from_address=local_addr),
        first,
        last,
    )


@export("mgt_local_clustering")
def mgt_local_clustering(
    offsets_addr: Int, neighbors_addr: Int, triangle_addr: Int, local_addr: Int, n: Int,
) abi("C") -> Float64:
    var offsets = IPtr(unsafe_from_address=offsets_addr)
    var neighbors = IPtr(unsafe_from_address=neighbors_addr)
    var triangle = IPtr(unsafe_from_address=triangle_addr)
    var local = FPtr(unsafe_from_address=local_addr)
    count_clustering_range(offsets, neighbors, triangle, local, 0, n)

    var closed = 0.0
    var wedges = 0.0
    for vertex in range(n):
        var start = Int(offsets[vertex])
        var stop = Int(offsets[vertex + 1])
        var degree = stop - start
        closed += Float64(triangle[vertex])
        wedges += Float64(degree * (degree - 1) // 2)
    return closed / wedges if wedges > 0.0 else 0.0
