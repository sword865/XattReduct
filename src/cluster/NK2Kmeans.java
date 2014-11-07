package cluster;

import weka.clusterers.NumberOfClustersRequestable;
import weka.clusterers.RandomizableClusterer;
import weka.core.Instances;
import weka.core.WeightedInstancesHandler;

public class NK2Kmeans extends  RandomizableClusterer  implements NumberOfClustersRequestable,WeightedInstancesHandler{

	private static final long serialVersionUID = 1L;

	@Override
	public void setNumClusters(int numClusters) throws Exception {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void buildClusterer(Instances data) throws Exception {
		// TODO Auto-generated method stub
		
	}

	@Override
	public int numberOfClusters() throws Exception {
		// TODO Auto-generated method stub
		return 0;
	}
	
	
	
}
