#include <stdio.h>
#include "graph.h"

extern "C" __declspec(dllexport)
void calculateSeam(int r, int c, float* pscore, float* plabel);

void calculateSeam(int r, int c, float* pscore, float* plabel)
{
	int node_num = r * c;
	int edge_num = (r - 1)*c + r * (c - 1);
	typedef Graph<float, float, float> GraphType;
	GraphType *g = new GraphType(node_num, edge_num);

	for (int i = 0; i < node_num; i++)
		g->add_node();

	for (int y = 0; y < r; y++)
	{
		for (int x = 0; x < c; x++)
		{
			g->add_tweights(x + y * c,
				plabel[x + y * c] == 1 ? 1 : 0,
				plabel[x + y * c] == 2 ? 1 : 0);

			if (x < c - 1)
			{
				float cap = 1e-8 + (pscore[x + y * c] + pscore[x + 1 + y * c])/2;
				g->add_edge(x + y * c, x + 1 + y * c, cap, cap);
			}

			if (y < r - 1)
			{
				float cap = 1e-8 + (pscore[x + y * c] + pscore[x + (y + 1) * c])/2;
				g->add_edge(x + y * c, x + (y + 1) * c, cap, cap);
			}
		}
	}

	float flow = g->maxflow();

	for (int y = 0; y < r; y++)
	{
		for (int x = 0; x < c; x++)
		{
			plabel[x + y * c] = (g->what_segment(x + y * c) == GraphType::SOURCE ? 1 : 2);
		}
	}

	delete g;
}
