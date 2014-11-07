package cluster.ABC;

import java.util.Random;

import weka.clusterers.NumberOfClustersRequestable;
import weka.clusterers.RandomizableClusterer;
import weka.core.DistanceFunction;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.WeightedInstancesHandler;

public class KernelKmeans  extends RandomizableClusterer  implements NumberOfClustersRequestable,WeightedInstancesHandler{
	
	protected static KernelDistances m_DistanceFunction = new KernelDistances();
	Random random=new Random();
	protected int m_NumClusters = 2;
	protected int maxiter=500;
	protected Instances m_ClusterCentroids;
//	protected Vector<Double> optvalueseq;
	protected int[] m_clusters;	
	
	public void setKernelpar(double par){
		m_DistanceFunction.setpar(par);
	}

	@Override
	public void buildClusterer(Instances data) throws Exception {
		// TODO Auto-generated method stub
		m_DistanceFunction.setInstances(data);
		int itercount=0;
		random=new Random(m_Seed);
		Instances centres=randomCentres(data,m_NumClusters,random);
		m_clusters=new int[data.numInstances()];
		double prefunvalue=-1;
		double optfunvalue=prefunvalue;		
		for(int iter=0;iter<maxiter;iter++){
			double [] clusterproperty=new double[centres.numInstances()];
			itercount++;
			for(int i=0;i<data.numInstances();i++){
				double[] dis=new double[centres.numInstances()];
				for(int j=0;j<dis.length;j++){
					if(iter==0){
						dis[j]=m_DistanceFunction.distance(data.instance(i),centres.instance(j));						
					}else{
						double kxx=m_DistanceFunction.getsimilarity(data.instance(i), data.instance(i));
						double tmp=0;double count=0;
						for(int k=0;k<m_clusters.length;k++){							
							if(m_clusters[k]==j){
								count++;
								tmp+=m_DistanceFunction.getsimilarity(data.instance(i), data.instance(k));
							}
						}
						tmp/=count;
						dis[j]=kxx-2*tmp+clusterproperty[j];
					}
				}
				double mindis=dis[0];int sel=0;		
				for(int j=1;j<dis.length;j++){
					if(dis[j]<mindis){
						sel=j;
						mindis=dis[j];
						}
				}
				m_clusters[i]=sel;
			}
			for(int i=0;i<clusterproperty.length;i++){
				Instances curcluster=new Instances(data,0);
				for(int j=0;j<data.numInstances();j++){
					if(m_clusters[j]==i){
						curcluster.add(data.instance(i));
					}
				}
				clusterproperty[i]=0;
				for(int j=0;j<curcluster.numInstances();j++){
					for(int k=0;k<curcluster.numInstances();k++){
						clusterproperty[i]+=m_DistanceFunction.getsimilarity(
								curcluster.instance(j),curcluster.instance(k));
					}
				}
				clusterproperty[i]/=(curcluster.numInstances()*curcluster.numInstances());
			}
			if(prefunvalue>-1 && optfunvalue-prefunvalue > -1e-8 )
				break;
			prefunvalue=optfunvalue;
		}
	}

	public Instance randomInstance(Instances data,Random random){
		Instance tmp=new Instance(data.numAttributes());
		for(int j=0;j<data.numAttributes();j++){
			int temp=random.nextInt(data.numInstances());
			double randvalue=data.instance(temp).value(j);
			tmp.setValue(j, randvalue);
		}
		return tmp;
	}
	
	public Instances randomCentres(Instances data,int K,Random random){
		Instances centres=new Instances(data,K);
		double[] datavalue;
		for(int i=0;i<K;i++){
			Instance tmp=randomInstance(data,random);
			centres.add(tmp);
		}
		return centres;
	}
	
	@Override
	public int numberOfClusters() throws Exception {
		// TODO Auto-generated method stub
		return m_NumClusters;
	}
	@Override
	public void setNumClusters(int numClusters) throws Exception {
		// TODO Auto-generated method stub
		m_NumClusters=numClusters;
	}

	public int[] getfinalcluster(){
		return m_clusters;
	}

}
