#include <iostream>
#include <lemon/list_graph.h>
#include <lemon/preflow.h>
#include <limits.h>
#include <stdlib.h>
#include <time.h>

using namespace lemon;

int main(int argc, char **argv)
{
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <width>" << std::endl;
    exit(1);
  }
  int width = atoi(argv[1]);
  int depth = width / 2;
  srand(time(NULL));

  ListDigraph g;

  auto s = g.addNode();
  auto t = g.addNode();

  std::vector<std::vector<ListDigraph::Node> > node_mat;
  lemon::ListDigraph::ArcMap<int> cap(g);

  for (int z = 0; z < depth; z++) {
    std::vector<ListDigraph::Node> node_row;
    for (int y = 0; y < width - 2*z; y++) {
      if (y <= width - 2*z) {
        // Create a node
        auto node = g.addNode();
        node_row.push_back(node);

        // Determine value, and link to source or target
        auto value = (rand() - RAND_MAX / 2) / (RAND_MAX / 100);

        if (value > 0) {
          cap[g.addArc(s, node)] = value;
        } else {
          cap[g.addArc(node, t)] = -value;
        }

        // Link to nodes above
        if (z > 0) {
          cap[g.addArc(node, node_mat[z-1][y])] = INT_MAX;
          cap[g.addArc(node, node_mat[z-1][y+1])] = INT_MAX;
          cap[g.addArc(node, node_mat[z-1][y+2])] = INT_MAX;
        }
      }
    }
    node_mat.push_back(node_row);
  }

  // Now call the min cut solver?
  auto preflower = Preflow<lemon::ListDigraph>(g, cap, s, t);
  preflower.runMinCut();

  int n = 0;
  for (auto row = node_mat.begin(); row != node_mat.end(); ++row) {
    // Shift the row over a bunch.
    for (int i = 0; i < n; i++)
      std::cout << "X";

    for (auto node = row->begin(); node != row->end(); ++node) {
      if (preflower.minCut(*node)) {
        std::cout << " ";
      } else {
        std::cout << "X";
      }
    }
    for (int i = 0; i < n; i++)
      std::cout << "X";
    n++;
    std::cout << std::endl;
  }

  return 0;
}
