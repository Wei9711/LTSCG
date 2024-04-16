#include <iostream>
#include <cstdio>
#include <algorithm>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <algorithm>
using namespace std;

const double INF = 1e30;
const double eps = 1e-6;

const int MaxN = 50; // number of agents
const int MaxM = 2500; // number of edges

template <class T>
inline void tension(T &a, const T &b)
{
    if (b < a)
        a = b;
}
template <class T>
inline void relax(T &a, const T &b)
{
    if (b > a)
        a = b;
}
template <class T>
inline int size(const T &a)
{
    return (int)a.size();
}

const int MaxNX = MaxN + MaxN;

struct edge
{
    int v, u;
    double w;

    edge() {}
    edge(const int &_v, const int &_u, const double &_w)
        : v(_v), u(_u), w(_w) {}
};

class Blossom_Solver
{
    int n, m;
    edge mat[MaxNX + 1][MaxNX + 1];

    int n_matches;
    double tot_weight;
    int mate[MaxNX + 1];
    double lab[MaxNX + 1];

    int q_n, q[MaxN];
    int fa[MaxNX + 1], col[MaxNX + 1];
    int slackv[MaxNX + 1];

    int n_x;
    int bel[MaxNX + 1], blofrom[MaxNX + 1][MaxNX + 1];
    vector<int> bloch[MaxNX + 1];

    inline double e_delta(const edge &e) // does not work inside blossoms
    {
        return lab[e.v] + lab[e.u] - mat[e.v][e.u].w * 2;
    }
    inline void update_slackv(int v, int x)
    {
        if (!slackv[x] || e_delta(mat[v][x]) < e_delta(mat[slackv[x]][x]))
            slackv[x] = v;
    }
    inline void calc_slackv(int x)
    {
        slackv[x] = 0;
        for (int v = 1; v <= n; v++)
            if (mat[v][x].w > 0 && bel[v] != x && col[bel[v]] == 0)
                update_slackv(v, x);
    }

    inline void q_push(int x)
    {
        if (x <= n)
            q[q_n++] = x;
        else
        {
            for (int i = 0; i < size(bloch[x]); i++)
                q_push(bloch[x][i]);
        }
    }
    inline void set_mate(int xv, int xu)
    {
        mate[xv] = mat[xv][xu].u;
        if (xv > n)
        {
            edge e = mat[xv][xu];
            int xr = blofrom[xv][e.v];
            int pr = find(bloch[xv].begin(), bloch[xv].end(), xr) - bloch[xv].begin();
            if (pr % 2 == 1)
            {
                reverse(bloch[xv].begin() + 1, bloch[xv].end());
                pr = size(bloch[xv]) - pr;
            }

            for (int i = 0; i < pr; i++)
                set_mate(bloch[xv][i], bloch[xv][i ^ 1]);
            set_mate(xr, xu);

            rotate(bloch[xv].begin(), bloch[xv].begin() + pr, bloch[xv].end());
        }
    }
    inline void set_bel(int x, int b)
    {
        bel[x] = b;
        if (x > n)
        {
            for (int i = 0; i < size(bloch[x]); i++)
                set_bel(bloch[x][i], b);
        }
    }

    inline void augment(int xv, int xu)
    {
        while (true)
        {
            int xnu = bel[mate[xv]];
            set_mate(xv, xu);
            if (!xnu)
                return;
            set_mate(xnu, bel[fa[xnu]]);
            xv = bel[fa[xnu]], xu = xnu;
        }
    }
    inline int get_lca(int xv, int xu)
    {
        static bool book[MaxNX + 1];
        for (int x = 1; x <= n_x; x++)
            book[x] = false;
        while (xv || xu)
        {
            if (xv)
            {
                if (book[xv])
                    return xv;
                book[xv] = true;
                xv = bel[mate[xv]];
                if (xv)
                    xv = bel[fa[xv]];
            }
            swap(xv, xu);
        }
        return 0;
    }

    inline void add_blossom(int xv, int xa, int xu)
    {
        int b = n + 1;
        while (b <= n_x && bel[b])
            b++;
        if (b > n_x)
            n_x++;

        lab[b] = 0;
        col[b] = 0;

        mate[b] = mate[xa];

        bloch[b].clear();
        bloch[b].push_back(xa);
        for (int x = xv; x != xa; x = bel[fa[bel[mate[x]]]])
            bloch[b].push_back(x), bloch[b].push_back(bel[mate[x]]), q_push(bel[mate[x]]);
        reverse(bloch[b].begin() + 1, bloch[b].end());
        for (int x = xu; x != xa; x = bel[fa[bel[mate[x]]]])
            bloch[b].push_back(x), bloch[b].push_back(bel[mate[x]]), q_push(bel[mate[x]]);

        set_bel(b, b);

        for (int x = 1; x <= n_x; x++)
        {
            mat[b][x].w = mat[x][b].w = 0;
            blofrom[b][x] = 0;
        }
        for (int i = 0; i < size(bloch[b]); i++)
        {
            int xs = bloch[b][i];
            for (int x = 1; x <= n_x; x++)
                if (mat[b][x].w == 0 || e_delta(mat[xs][x]) < e_delta(mat[b][x]))
                    mat[b][x] = mat[xs][x], mat[x][b] = mat[x][xs];
            for (int x = 1; x <= n_x; x++)
                if (blofrom[xs][x])
                    blofrom[b][x] = xs;
        }
        calc_slackv(b);
    }
    inline void expand_blossom1(int b) // lab[b] == 1
    {
        for (int i = 0; i < size(bloch[b]); i++)
            set_bel(bloch[b][i], bloch[b][i]);

        int xr = blofrom[b][mat[b][fa[b]].v];
        int pr = find(bloch[b].begin(), bloch[b].end(), xr) - bloch[b].begin();
        if (pr % 2 == 1)
        {
            reverse(bloch[b].begin() + 1, bloch[b].end());
            pr = size(bloch[b]) - pr;
        }

        for (int i = 0; i < pr; i += 2)
        {
            int xs = bloch[b][i], xns = bloch[b][i + 1];
            fa[xs] = mat[xns][xs].v;
            col[xs] = 1, col[xns] = 0;
            slackv[xs] = 0, calc_slackv(xns);
            q_push(xns);
        }
        col[xr] = 1;
        fa[xr] = fa[b];
        for (int i = pr + 1; i < size(bloch[b]); i++)
        {
            int xs = bloch[b][i];
            col[xs] = -1;
            calc_slackv(xs);
        }

        bel[b] = 0;
    }
    inline void expand_blossom_final(int b) // at the final stage
    {
        for (int i = 0; i < size(bloch[b]); i++)
        {
            if (bloch[b][i] > n && fabs(lab[bloch[b][i]]) < eps)
                expand_blossom_final(bloch[b][i]);
            else
                set_bel(bloch[b][i], bloch[b][i]);
        }
        bel[b] = 0;
    }

    inline bool on_found_edge(const edge &e)
    {
        int xv = bel[e.v], xu = bel[e.u];
        if (col[xu] == -1)
        {
            int nv = bel[mate[xu]];
            fa[xu] = e.v;
            col[xu] = 1, col[nv] = 0;
            slackv[xu] = slackv[nv] = 0;
            q_push(nv);
        }
        else if (col[xu] == 0)
        {
            int xa = get_lca(xv, xu);
            if (!xa)
            {
                augment(xv, xu), augment(xu, xv);
                for (int b = n + 1; b <= n_x; b++)
                    if (bel[b] == b && fabs(lab[b]) < eps)
                        expand_blossom_final(b);
                return true;
            }
            else
                add_blossom(xv, xa, xu);
        }
        return false;
    }

    bool match()
    {
        for (int x = 1; x <= n_x; x++)
            col[x] = -1, slackv[x] = 0;

        q_n = 0;
        for (int x = 1; x <= n_x; x++)
            if (bel[x] == x && !mate[x])
                fa[x] = 0, col[x] = 0, slackv[x] = 0, q_push(x);
        if (q_n == 0)
            return false;

        while (true)
        {
            for (int i = 0; i < q_n; i++)
            {
                int v = q[i];
                for (int u = 1; u <= n; u++)
                    if (mat[v][u].w > 0 && bel[v] != bel[u])
                    {
                        double d = e_delta(mat[v][u]);
                        if (fabs(d) < eps)
                        {
                            if (on_found_edge(mat[v][u]))
                                return true;
                        }
                        else if (col[bel[u]] == -1 || col[bel[u]] == 0)
                            update_slackv(v, bel[u]);
                    }
            }

            double d = INF;
            for (int v = 1; v <= n; v++)
                if (col[bel[v]] == 0)
                    tension(d, lab[v]);
            for (int b = n + 1; b <= n_x; b++)
                if (bel[b] == b && col[b] == 1)
                    tension(d, lab[b] / 2);
            for (int x = 1; x <= n_x; x++)
                if (bel[x] == x && slackv[x])
                {
                    if (col[x] == -1)
                        tension(d, e_delta(mat[slackv[x]][x]));
                    else if (col[x] == 0)
                        tension(d, e_delta(mat[slackv[x]][x]) / 2);
                }

            for (int v = 1; v <= n; v++)
            {
                if (col[bel[v]] == 0)
                    lab[v] -= d;
                else if (col[bel[v]] == 1)
                    lab[v] += d;
            }
            for (int b = n + 1; b <= n_x; b++)
                if (bel[b] == b)
                {
                    if (col[bel[b]] == 0)
                        lab[b] += d * 2;
                    else if (col[bel[b]] == 1)
                        lab[b] -= d * 2;
                }

            q_n = 0;
            for (int v = 1; v <= n; v++)
                if (fabs(lab[v]) < eps) // all unmatched vertices' labels are zero! cheers!
                    return false;
            for (int x = 1; x <= n_x; x++)
                if (bel[x] == x && slackv[x] && bel[slackv[x]] != x && fabs(e_delta(mat[slackv[x]][x])) < eps)
                {
                    if (on_found_edge(mat[slackv[x]][x]))
                        return true;
                }
            for (int b = n + 1; b <= n_x; b++)
                if (bel[b] == b && col[b] == 1 && fabs(lab[b]) < eps)
                    expand_blossom1(b);
        }
        return false;
    }

    void calc_max_weight_match()
    {
        for (int v = 1; v <= n; v++)
            mate[v] = 0;

        n_x = n;
        n_matches = 0;
        tot_weight = 0;

        bel[0] = 0;
        for (int v = 1; v <= n; v++)
            bel[v] = v, bloch[v].clear();
        for (int v = 1; v <= n; v++)
            for (int u = 1; u <= n; u++)
                blofrom[v][u] = v == u ? v : 0;

        double w_max = 0;
        for (int v = 1; v <= n; v++)
            for (int u = 1; u <= n; u++)
                relax(w_max, mat[v][u].w);
        for (int v = 1; v <= n; v++)
            lab[v] = w_max;

        while (match())
            n_matches++;

        for (int v = 1; v <= n; v++)
            if (mate[v] && mate[v] < v)
                tot_weight += mat[v][mate[v]].w;
    }

    public:
    void solve(double *py_f, double *best_graph, int py_n)
    {
        n = py_n;
        for (int v = 1; v <= n; v++)
            for (int u = 1; u <= n; u++)
                mat[v][u] = edge(v, u, py_f[(v - 1) * n + (u - 1)]);
        /*n = getint(), m = getint();

    for (int v = 1; v <= n; v++)
        for (int u = 1; u <= n; u++)
            mat[v][u] = edge(v, u, 0);

    for (int i = 0; i < m; i++)
    {
        int v = getint(), u = getint(), w = getint();
        mat[v][u].w = mat[u][v].w = w;
    }*/

        calc_max_weight_match();

        for (int v = 0; v < n; v++)
            for (int u = 0; u < n; u++)
                best_graph[v * n + u] = 0;
        for (int v = 1; v <= n; v++)
            if (mate[v] != 0)
                best_graph[(v - 1) * n + (mate[v] - 1)] = 1;
        /*printf("%lld\n", tot_weight);
    for (int v = 1; v <= n; v++)
        printf("%d ", mate[v]);
    printf("\n");*/
    }
};

const int MAX_BATCH_SIZE = 50;
Blossom_Solver blossom_solver[MAX_BATCH_SIZE];

extern "C" void
blossom_solve_para(double *py_f, double *best_graph, int py_bs, int py_n)
{
#pragma omp parallel for schedule(dynamic, 1) num_threads(MAX_BATCH_SIZE)
    for (int i = 0; i < py_bs; i++)
        blossom_solver[i].solve(py_f + i * py_n * py_n, best_graph + i * py_n * py_n, py_n);
}

class GraphEpsilonGreedy_solver
{
    int n, m, a[MaxN];
    bool flag;

    public:
    void solve(double *graph, int py_n, double eps)
    {
        n = py_n, m = 0;
        if (!flag)
            srand(time(0)), flag = true;
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j)
                if (graph[i * n + j] > 0.5)
                    if (double(rand()) / RAND_MAX < eps)
                    {
                        graph[i * n + j] = graph[j * n + i] = 0;
                        a[++m] = i, a[++m] = j;
                    }
        if (m == 2 && n >= 4)
        {
            int x = rand() % n, y = rand() % n;
            while (graph[x * n + y] < 0.5)
                x = rand() % n, y = rand() % n;
            graph[x * n + y] = graph[y * n + x] = 0;
            a[++m] = x, a[++m] = y;
        }
        random_shuffle(a + 1, a + n + 1);
        for (int i = 1; i <= m; i += 2)
            graph[a[i] * n + a[i + 1]] = graph[a[i + 1] * n + a[i]] = 1;
    }
};

GraphEpsilonGreedy_solver epsgreedy[MAX_BATCH_SIZE];

extern "C" void
graph_epsilon_greedy(double *graphs, int py_bs, int py_n, double eps)
{
#pragma omp parallel for schedule(dynamic, 1) num_threads(MAX_BATCH_SIZE)
    for (int i = 0; i < py_bs; i++)
        epsgreedy[i].solve(graphs + i * py_n * py_n, py_n, eps);
}