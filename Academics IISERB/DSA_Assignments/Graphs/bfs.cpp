// bfs can be implemented using the queues

class Solution {
	public: 
		//function to return Breadth First Traversal of given graph
		vector<int> bfsOfGraph(int V, vector<int> adj[]) {
			int vis[n] = {0};
			vis[0] = 1;
			queue<int> q;
			q.push(0);

			vector<int> bfs;
			while (!q.empty()) {
				int node = q.front();
				q.pop();
				bfs.push(node);

				for (auto it : adj[node]) {
					if (!vis[it]) {
						vis[it] =1;
						q.push(it);
					}
				}
			}
			return bfs;
		}
	};
	
