// Compiled C++20 Multi-Layer 45-Degree Octilinear A* Kernel for PCB World Model.
// Exposed via C ABI for zero-copy NumPy pointer invocation through ctypes.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

constexpr float kSqrt2 = 1.4142135623730950488f;
constexpr float kDiagDelta = kSqrt2 - 1.0f;
constexpr float kBendPenalty = 0.06f;

struct DirStep {
    int32_t dx;
    int32_t dy;
    float cost;
};

constexpr DirStep kOctilinearDirs[8] = {
    {1, 0, 1.0f},
    {-1, 0, 1.0f},
    {0, 1, 1.0f},
    {0, -1, 1.0f},
    {1, 1, kSqrt2},
    {1, -1, kSqrt2},
    {-1, 1, kSqrt2},
    {-1, -1, kSqrt2},
};

struct HeapNode {
    float f;
    float g;
    int32_t idx;
};

struct MinHeapCmp {
    bool operator()(const HeapNode& a, const HeapNode& b) const noexcept {
        if (a.f != b.f) {
            return a.f > b.f;
        }
        if (a.g != b.g) {
            return a.g > b.g;
        }
        return a.idx > b.idx;
    }
};

// Thread-local reusable buffers eliminate repeated heap allocations across MST segments.
struct AStarWorkspace {
    std::vector<float> h_map;
    std::vector<float> g_score;
    std::vector<int32_t> parent;
    std::vector<uint8_t> closed;
    std::vector<uint8_t> is_target;
    std::vector<int32_t> rev_path;
    std::vector<HeapNode> heap;

    void prepare(std::size_t hw, std::size_t total_nodes) {
        const float inf = std::numeric_limits<float>::infinity();
        h_map.assign(hw, inf);
        g_score.assign(total_nodes, inf);
        parent.assign(total_nodes, -1);
        closed.assign(total_nodes, 0);
        is_target.assign(total_nodes, 0);
        rev_path.clear();
        heap.clear();
        if (heap.capacity() < 4096) {
            heap.reserve(4096);
        }
    }
};

thread_local AStarWorkspace g_workspace;

int32_t run_astar_multilayer_impl(
    int32_t L,
    int32_t H,
    int32_t W,
    const int32_t* start_nodes,
    int32_t num_starts,
    const int32_t* target_nodes,
    int32_t num_targets,
    const uint8_t* blocked,
    const uint8_t* via_blocked,
    const uint8_t* stagger_blocked,
    const float* soft_penalty,
    const float* cost_field,
    const int32_t* occ_layers,
    const uint8_t* allowed_layers,
    const float* transition_costs,
    int32_t net_id,
    float min_stagger_cells,
    int32_t max_expansions,
    int32_t* out_path,
    int32_t max_path_len,
    int32_t* out_path_len
) noexcept {
    if (L <= 0 || H <= 0 || W <= 0 || blocked == nullptr || via_blocked == nullptr ||
        soft_penalty == nullptr || cost_field == nullptr || occ_layers == nullptr ||
        allowed_layers == nullptr || transition_costs == nullptr || out_path == nullptr ||
        out_path_len == nullptr || max_path_len <= 0) {
        return -1;
    }
    *out_path_len = 0;
    if (num_starts <= 0 || num_targets <= 0 || start_nodes == nullptr || target_nodes == nullptr) {
        return 0;
    }

    const std::size_t hw = static_cast<std::size_t>(H) * static_cast<std::size_t>(W);
    const std::size_t total_nodes = static_cast<std::size_t>(L) * hw;

    AStarWorkspace& ws = g_workspace;
    ws.prepare(hw, total_nodes);

    // Mark valid target cells and collect unique (tx, ty) coordinates for admissible heuristic.
    std::vector<int32_t> unique_xy;
    unique_xy.reserve(static_cast<std::size_t>(num_targets) * 2);

    bool has_valid_target = false;
    for (int32_t i = 0; i < num_targets; ++i) {
        const int32_t tl = target_nodes[i * 3 + 0];
        const int32_t tx = target_nodes[i * 3 + 1];
        const int32_t ty = target_nodes[i * 3 + 2];
        if (tl < 0 || tl >= L || tx < 0 || tx >= W || ty < 0 || ty >= H) {
            continue;
        }
        const std::size_t tidx =
            static_cast<std::size_t>(tl) * hw + static_cast<std::size_t>(ty) * W + tx;
        ws.is_target[tidx] = 1;
        has_valid_target = true;

        bool seen = false;
        for (std::size_t k = 0; k < unique_xy.size(); k += 2) {
            if (unique_xy[k] == tx && unique_xy[k + 1] == ty) {
                seen = true;
                break;
            }
        }
        if (!seen) {
            unique_xy.push_back(tx);
            unique_xy.push_back(ty);
        }
    }

    if (!has_valid_target) {
        return 0;
    }

    // Build 2D admissible octilinear distance heuristic across the board.
    const std::size_t num_unique = unique_xy.size() / 2;
    for (int32_t y = 0; y < H; ++y) {
        const std::size_t row_off = static_cast<std::size_t>(y) * static_cast<std::size_t>(W);
        for (int32_t x = 0; x < W; ++x) {
            float best = std::numeric_limits<float>::infinity();
            for (std::size_t k = 0; k < num_unique; ++k) {
                const int32_t dx = std::abs(unique_xy[2 * k] - x);
                const int32_t dy = std::abs(unique_xy[2 * k + 1] - y);
                const int32_t hi = std::max(dx, dy);
                const int32_t lo = std::min(dx, dy);
                const float d = static_cast<float>(hi) + kDiagDelta * static_cast<float>(lo);
                if (d < best) {
                    best = d;
                }
            }
            ws.h_map[row_off + static_cast<std::size_t>(x)] = best;
        }
    }

    MinHeapCmp cmp;
    for (int32_t i = 0; i < num_starts; ++i) {
        const int32_t sl = start_nodes[i * 3 + 0];
        const int32_t sx = start_nodes[i * 3 + 1];
        const int32_t sy = start_nodes[i * 3 + 2];
        if (sl < 0 || sl >= L || sx < 0 || sx >= W || sy < 0 || sy >= H) {
            continue;
        }
        const int32_t sidx = static_cast<int32_t>(
            static_cast<std::size_t>(sl) * hw + static_cast<std::size_t>(sy) * W + sx
        );
        if (ws.g_score[static_cast<std::size_t>(sidx)] > 0.0f) {
            ws.g_score[static_cast<std::size_t>(sidx)] = 0.0f;
            const float h0 = ws.h_map[static_cast<std::size_t>(sy) * W + sx];
            ws.heap.push_back(HeapNode{h0, 0.0f, sidx});
            std::push_heap(ws.heap.begin(), ws.heap.end(), cmp);
        }
    }

    int32_t expansions = 0;
    const int32_t hw_i = static_cast<int32_t>(hw);

    while (!ws.heap.empty() && expansions < max_expansions) {
        std::pop_heap(ws.heap.begin(), ws.heap.end(), cmp);
        const HeapNode top = ws.heap.back();
        ws.heap.pop_back();

        const int32_t idx = top.idx;
        const std::size_t uidx = static_cast<std::size_t>(idx);
        if (ws.closed[uidx] != 0) {
            continue;
        }
        ws.closed[uidx] = 1;
        ++expansions;

        if (ws.is_target[uidx] != 0) {
            ws.rev_path.clear();
            int32_t cur = idx;
            while (cur != -1) {
                const int32_t p = ws.parent[static_cast<std::size_t>(cur)];
                ws.rev_path.push_back(cur);
                if (p != -1) {
                    const int32_t cl = cur / hw_i;
                    const int32_t pl = p / hw_i;
                    if (std::abs(cl - pl) == 2 && allowed_layers[(cl + pl) / 2] == 0) {
                        const int32_t mid_l = (cl + pl) / 2;
                        const int32_t crem = cur % hw_i;
                        const int32_t prem = p % hw_i;
                        const int32_t cx = crem % W;
                        const int32_t cy = crem / W;
                        const int32_t px = prem % W;
                        const int32_t py = prem / W;
                        const int32_t dx = px - cx;
                        const int32_t dy = py - cy;
                        const int32_t steps = std::max(std::abs(dx), std::abs(dy));
                        const int32_t ux = (dx > 0) ? 1 : ((dx < 0) ? -1 : 0);
                        const int32_t uy = (dy > 0) ? 1 : ((dy < 0) ? -1 : 0);
                        for (int32_t s = 0; s <= steps; ++s) {
                            const int32_t mx = cx + s * ux;
                            const int32_t my = cy + s * uy;
                            const int32_t m_node =
                                static_cast<int32_t>(static_cast<std::size_t>(mid_l) * hw +
                                                     static_cast<std::size_t>(my) * W + mx);
                            ws.rev_path.push_back(m_node);
                        }
                    }
                }
                cur = p;
            }
            const int32_t path_n = static_cast<int32_t>(ws.rev_path.size());
            if (path_n > max_path_len) {
                return -1;
            }
            for (int32_t p = 0; p < path_n; ++p) {
                const int32_t node = ws.rev_path[static_cast<std::size_t>(path_n - 1 - p)];
                const int32_t cl = node / hw_i;
                const int32_t crem = node % hw_i;
                const int32_t cy = crem / W;
                const int32_t cx = crem % W;
                out_path[p * 3 + 0] = cl;
                out_path[p * 3 + 1] = cx;
                out_path[p * 3 + 2] = cy;
            }
            *out_path_len = path_n;
            return 1;
        }

        const float cur_g = top.g;
        const int32_t l = idx / hw_i;
        const int32_t rem = idx % hw_i;
        const int32_t y = rem / W;
        const int32_t x = rem % W;
        const std::size_t layer_off = static_cast<std::size_t>(l) * hw;

        const int32_t p_idx = ws.parent[uidx];
        int32_t pl = -1;
        int32_t prev_dx = 0;
        int32_t prev_dy = 0;
        if (p_idx != -1) {
            pl = p_idx / hw_i;
            if (pl == l) {
                const int32_t prem = p_idx % hw_i;
                const int32_t py = prem / W;
                const int32_t px = prem % W;
                prev_dx = x - px;
                prev_dy = y - py;
            }
        }

        // 1. Expand 8 planar octilinear neighbors when planar routing is permitted on layer l.
        if (allowed_layers[l] != 0) {
            for (const auto& step : kOctilinearDirs) {
                const int32_t nx = x + step.dx;
                const int32_t ny = y + step.dy;
                if (nx < 0 || nx >= W || ny < 0 || ny >= H) {
                    continue;
                }
                const std::size_t n_xy = static_cast<std::size_t>(ny) * W + nx;
                const std::size_t n_uidx = layer_off + n_xy;
                if (ws.closed[n_uidx] != 0) {
                    continue;
                }
                const bool target_hit = (ws.is_target[n_uidx] != 0);
                if (!target_hit && blocked[n_uidx] != 0) {
                    continue;
                }

                // Prevent diagonal corner cutting & diagonal X-crossing through foreign traces.
                if (step.dx != 0 && step.dy != 0 && !target_hit) {
                    const std::size_t side_a = layer_off + static_cast<std::size_t>(y) * W + nx;
                    const std::size_t side_b = layer_off + static_cast<std::size_t>(ny) * W + x;
                    if (blocked[side_a] != 0 && blocked[side_b] != 0) {
                        continue;
                    }
                    const int32_t occ_a = occ_layers[side_a];
                    const int32_t occ_b = occ_layers[side_b];
                    if (occ_a > 0 && occ_a != net_id && occ_b > 0 && occ_b != net_id) {
                        continue;
                    }
                }

                const float bend_pen =
                    (p_idx != -1 && pl == l && (step.dx != prev_dx || step.dy != prev_dy))
                        ? kBendPenalty
                        : 0.0f;
                const float cell_cost = cost_field[n_uidx] + soft_penalty[n_uidx];
                const float tentative_g = cur_g + step.cost * (1.0f + cell_cost) + bend_pen;

                if (tentative_g < ws.g_score[n_uidx]) {
                    ws.g_score[n_uidx] = tentative_g;
                    ws.parent[n_uidx] = idx;
                    const float h = ws.h_map[n_xy];
                    ws.heap.push_back(HeapNode{tentative_g + h, tentative_g, static_cast<int32_t>(n_uidx)});
                    std::push_heap(ws.heap.begin(), ws.heap.end(), cmp);
                }
            }
        }

        // 2. Expand inter-layer via transitions (l -> next_l) checking via clearance across span [lo..hi].
        const std::size_t cell_xy = static_cast<std::size_t>(y) * W + x;
        for (int32_t next_l = 0; next_l < L; ++next_l) {
            if (next_l == l) {
                continue;
            }
            const float t_cost = transition_costs[static_cast<std::size_t>(l) * L + next_l];
            if (t_cost < 0.0f || !std::isfinite(t_cost)) {
                continue;
            }
            const int32_t lo = std::min(l, next_l);
            const int32_t hi = std::max(l, next_l);

            // Rev 12: Staggered-only microvia reliability lockout across adjacent buildup layers.
            if (min_stagger_cells > 0.0f && (hi - lo) == 1) {
                if (stagger_blocked != nullptr) {
                    const int32_t iface = lo;
                    if (stagger_blocked[static_cast<std::size_t>(iface) * hw + cell_xy] != 0) {
                        continue;
                    }
                }
                const float min_stag_sq = min_stagger_cells * min_stagger_cells - 1e-4f;
                int32_t cur = idx;
                int32_t hops = 0;
                bool stagger_violated = false;
                while (cur != -1 && hops < 64) {
                    const int32_t p = ws.parent[static_cast<std::size_t>(cur)];
                    if (p == -1) {
                        break;
                    }
                    const int32_t p_l = p / hw_i;
                    if (p_l != l) {
                        if (std::abs(p_l - l) == 1) {
                            const int32_t crem = cur % hw_i;
                            const int32_t cy = crem / W;
                            const int32_t cx = crem % W;
                            const float ddx = static_cast<float>(x - cx);
                            const float ddy = static_cast<float>(y - cy);
                            if (ddx * ddx + ddy * ddy < min_stag_sq) {
                                stagger_violated = true;
                            }
                        }
                        break;
                    }
                    cur = p;
                    ++hops;
                }
                if (stagger_violated) {
                    continue;
                }
            }

            bool via_ok = true;
            for (int32_t k = lo; k <= hi; ++k) {
                if (via_blocked[static_cast<std::size_t>(k) * hw + cell_xy] != 0) {
                    via_ok = false;
                    break;
                }
            }
            if (!via_ok) {
                continue;
            }

            const std::size_t n_uidx = static_cast<std::size_t>(next_l) * hw + cell_xy;
            if (ws.closed[n_uidx] != 0) {
                continue;
            }
            const float tentative_g = cur_g + t_cost + soft_penalty[n_uidx];
            if (tentative_g < ws.g_score[n_uidx]) {
                ws.g_score[n_uidx] = tentative_g;
                ws.parent[n_uidx] = idx;
                const float h = ws.h_map[cell_xy];
                ws.heap.push_back(HeapNode{tentative_g + h, tentative_g, static_cast<int32_t>(n_uidx)});
                std::push_heap(ws.heap.begin(), ws.heap.end(), cmp);
            }
        }

        // 3. Atomic 2-via staggered microvia bridge across non-planar buildup layers (e.g., 0->1->2, 2->1->0, 5->6->7, 7->6->5).
        if (min_stagger_cells > 0.0f && allowed_layers[l] != 0) {
            const int32_t s_ortho = std::max(1, static_cast<int32_t>(std::ceil(min_stagger_cells)));
            const int32_t s_diag = std::max(1, static_cast<int32_t>(std::ceil(min_stagger_cells / kSqrt2)));
            for (int32_t dir_l : {-1, 1}) {
                const int32_t mid_l = l + dir_l;
                const int32_t next_l = l + 2 * dir_l;
                if (mid_l < 0 || mid_l >= L || next_l < 0 || next_l >= L) {
                    continue;
                }
                if (allowed_layers[mid_l] != 0 || allowed_layers[next_l] == 0) {
                    continue;
                }
                const float t1 = transition_costs[static_cast<std::size_t>(l) * L + mid_l];
                const float t2 = transition_costs[static_cast<std::size_t>(mid_l) * L + next_l];
                if (t1 < 0.0f || t2 < 0.0f || !std::isfinite(t1) || !std::isfinite(t2)) {
                    continue;
                }
                const int32_t lo1 = std::min(l, mid_l);
                const int32_t hi1 = std::max(l, mid_l);
                if (via_blocked[static_cast<std::size_t>(lo1) * hw + cell_xy] != 0 ||
                    via_blocked[static_cast<std::size_t>(hi1) * hw + cell_xy] != 0) {
                    continue;
                }
                if (stagger_blocked != nullptr &&
                    stagger_blocked[static_cast<std::size_t>(lo1) * hw + cell_xy] != 0) {
                    continue;
                }
                const std::size_t mid_off = static_cast<std::size_t>(mid_l) * hw;
                if (blocked[mid_off + cell_xy] != 0) {
                    continue;
                }
                const int32_t lo2 = std::min(mid_l, next_l);
                const int32_t hi2 = std::max(mid_l, next_l);

                for (const auto& step : kOctilinearDirs) {
                    const int32_t steps = (step.dx != 0 && step.dy != 0) ? s_diag : s_ortho;
                    const int32_t nx = x + step.dx * steps;
                    const int32_t ny = y + step.dy * steps;
                    if (nx < 0 || nx >= W || ny < 0 || ny >= H) {
                        continue;
                    }
                    const std::size_t n_xy = static_cast<std::size_t>(ny) * W + nx;
                    const std::size_t n_uidx = static_cast<std::size_t>(next_l) * hw + n_xy;
                    if (ws.closed[n_uidx] != 0) {
                        continue;
                    }
                    if (via_blocked[static_cast<std::size_t>(lo2) * hw + n_xy] != 0 ||
                        via_blocked[static_cast<std::size_t>(hi2) * hw + n_xy] != 0) {
                        continue;
                    }
                    if (stagger_blocked != nullptr &&
                        stagger_blocked[static_cast<std::size_t>(lo2) * hw + n_xy] != 0) {
                        continue;
                    }

                    bool dogbone_ok = true;
                    float dogbone_cost = 0.0f;
                    for (int32_t k = 1; k <= steps; ++k) {
                        const int32_t cx = x + step.dx * k;
                        const int32_t cy = y + step.dy * k;
                        const std::size_t c_xy = static_cast<std::size_t>(cy) * W + cx;
                        const std::size_t c_mid_uidx = mid_off + c_xy;
                        if (blocked[c_mid_uidx] != 0) {
                            dogbone_ok = false;
                            break;
                        }
                        if (step.dx != 0 && step.dy != 0) {
                            const int32_t px_k = cx - step.dx;
                            const int32_t py_k = cy - step.dy;
                            const std::size_t side_a = mid_off + static_cast<std::size_t>(py_k) * W + cx;
                            const std::size_t side_b = mid_off + static_cast<std::size_t>(cy) * W + px_k;
                            if (blocked[side_a] != 0 && blocked[side_b] != 0) {
                                dogbone_ok = false;
                                break;
                            }
                            const int32_t occ_a = occ_layers[side_a];
                            const int32_t occ_b = occ_layers[side_b];
                            if (occ_a > 0 && occ_a != net_id && occ_b > 0 && occ_b != net_id) {
                                dogbone_ok = false;
                                break;
                            }
                        }
                        dogbone_cost += step.cost * (1.0f + cost_field[c_mid_uidx] + soft_penalty[c_mid_uidx]);
                    }
                    if (!dogbone_ok) {
                        continue;
                    }

                    const float tentative_g = cur_g + t1 + t2 + dogbone_cost + soft_penalty[n_uidx];
                    if (tentative_g < ws.g_score[n_uidx]) {
                        ws.g_score[n_uidx] = tentative_g;
                        ws.parent[n_uidx] = idx;
                        const float h = ws.h_map[n_xy];
                        ws.heap.push_back(HeapNode{tentative_g + h, tentative_g, static_cast<int32_t>(n_uidx)});
                        std::push_heap(ws.heap.begin(), ws.heap.end(), cmp);
                    }
                }
            }
        }
    }

    return 0;
}

}  // namespace

extern "C" {

/// Returns ABI version (2) for runtime sanity verification.
int32_t pcb_astar_abi_version() noexcept {
    return 2;
}

/// Multi-layer 45-degree octilinear A* search.
int32_t astar_multilayer(
    int32_t L,
    int32_t H,
    int32_t W,
    const int32_t* start_nodes,
    int32_t num_starts,
    const int32_t* target_nodes,
    int32_t num_targets,
    const uint8_t* blocked,
    const uint8_t* via_blocked,
    const float* soft_penalty,
    const float* cost_field,
    const int32_t* occ_layers,
    const uint8_t* allowed_layers,
    const float* transition_costs,
    int32_t net_id,
    int32_t max_expansions,
    int32_t* out_path,
    int32_t max_path_len,
    int32_t* out_path_len
) noexcept {
    return run_astar_multilayer_impl(
        L, H, W, start_nodes, num_starts, target_nodes, num_targets,
        blocked, via_blocked, nullptr, soft_penalty, cost_field, occ_layers,
        allowed_layers, transition_costs, net_id, 0.0f, max_expansions,
        out_path, max_path_len, out_path_len
    );
}

/// Rev 12 HDI-aware multi-layer 45-degree octilinear A* with staggered-microvia lockout.
int32_t astar_multilayer_hdi(
    int32_t L,
    int32_t H,
    int32_t W,
    const int32_t* start_nodes,
    int32_t num_starts,
    const int32_t* target_nodes,
    int32_t num_targets,
    const uint8_t* blocked,
    const uint8_t* via_blocked,
    const uint8_t* stagger_blocked,
    const float* soft_penalty,
    const float* cost_field,
    const int32_t* occ_layers,
    const uint8_t* allowed_layers,
    const float* transition_costs,
    int32_t net_id,
    float min_stagger_cells,
    int32_t max_expansions,
    int32_t* out_path,
    int32_t max_path_len,
    int32_t* out_path_len
) noexcept {
    return run_astar_multilayer_impl(
        L, H, W, start_nodes, num_starts, target_nodes, num_targets,
        blocked, via_blocked, stagger_blocked, soft_penalty, cost_field, occ_layers,
        allowed_layers, transition_costs, net_id, min_stagger_cells, max_expansions,
        out_path, max_path_len, out_path_len
    );
}

}  // extern "C"
