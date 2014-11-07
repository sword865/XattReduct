package cluster.ABC;

import java.util.Random;
import java.util.Vector;

import cluster.ABC.SemiIndex.LinkPair;
import weka.clusterers.NumberOfClustersRequestable;
import weka.clusterers.RandomizableClusterer;
import weka.clusterers.SimpleKMeans;
import weka.core.DistanceFunction;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.WeightedInstancesHandler;

public class CopKmeans extends RandomizableClusterer  implements NumberOfClustersRequestable,WeightedInstancesHandler{
	Vector<LinkPair> mustlink=new Vector<LinkPair>();
	Vector<LinkPair> canotlink=new Vector<LinkPair>();		
	Random random=new Random();
	protected int m_NumClusters = 2;
	protected int maxiter=100;
	protected Instances m_ClusterCentroids;
	protected static DistanceFunction m_DistanceFunction = new EuclideanDistance();
//	protected Vector<Double> optvalueseq;
	protected int[] m_clusters;	
	
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
	
	
	public void setLink(Vector<LinkPair> inputmustlink,
			Vector<LinkPair> inputcanotlink) {
		// TODO Auto-generated constructor stub
		mustlink.clear();
		canotlink.clear();
		for(int i=0;i<inputmustlink.size();i++){
			mustlink.add(inputmustlink.elementAt(i));
		}
		for(int i=0;i<inputcanotlink.size();i++){
			canotlink.add(inputcanotlink.elementAt(i));
		}
	}
	
	public Instances getNewCentres(Instances data,int clusters[],int K){
		Instances[] memberSet=new Instances[K];
		for(int i=0;i<memberSet.length;i++){
			memberSet[i]=new Instances(data,0);
		}
		for(int i=0;i<data.numInstances();i++){
			memberSet[clusters[i]].add(data.instance(i));
		}
		Instances centres=new Instances(data,K);
		for(int i=0;i<K;i++){
			Instance tmp=new Instance(data.numAttributes());
			for(int j=0;j<data.numAttributes();j++){
				tmp.setValue(j, memberSet[i].meanOrMode(j));
			}
			centres.add(tmp);
		}		
		return centres;
	}
	
	boolean ViolateConstraint(int dataid,int[] m_clusters){
		for(int i=0;i<mustlink.size();i++){
			if((mustlink.elementAt(i).first==dataid&&
					mustlink.elementAt(i).second<dataid)||
					(mustlink.elementAt(i).first<dataid&&
							mustlink.elementAt(i).second==dataid)){
				if(m_clusters[mustlink.elementAt(i).first]!=
						m_clusters[mustlink.elementAt(i).second]){
					return true;
				}
			}
		}
		for(int i=0;i<canotlink.size();i++){
			if((canotlink.elementAt(i).first==dataid&&
					canotlink.elementAt(i).second<dataid)||
					(canotlink.elementAt(i).first<dataid&&
							canotlink.elementAt(i).second==dataid)){
				if(m_clusters[canotlink.elementAt(i).first]==
						m_clusters[canotlink.elementAt(i).second]){
					return true;
				}
			}
		}
		return false;
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
			itercount++;
			for(int i=0;i<data.numInstances();i++){
				double[] dis=new double[centres.numInstances()];
				for(int j=0;j<dis.length;j++){
					dis[j]=m_DistanceFunction.distance(data.instance(i),centres.instance(j));
				}
				while(true){
					double mindis=dis[0];int sel=0;		
					for(int j=1;j<dis.length;j++){
						if(dis[j]<mindis){
							sel=j;
							mindis=dis[j];
						}
					}
					if(mindis==Double.POSITIVE_INFINITY){
						System.out.printf("Can not give a result\n");
						return;
					}
					m_clusters[i]=sel;
					if(ViolateConstraint(i,m_clusters)){
						dis[sel]=Double.POSITIVE_INFINITY;
					}else{
						break;
					}
				}
			}
			centres=getNewCentres(data,m_clusters,m_NumClusters);
			//optfunvalue=optfun(data,centres,m_clusters);
			//optvalueseq.add(optfunvalue);			
			if(prefunvalue>-1 && optfunvalue-prefunvalue > -1e-8 )
				break;
			prefunvalue=optfunvalue;
		}
		m_ClusterCentroids=new Instances(centres);
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
