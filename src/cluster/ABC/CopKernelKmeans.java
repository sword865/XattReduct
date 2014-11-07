package cluster.ABC;

import java.util.Random;
import java.util.Vector;

import weka.core.Instances;

import cluster.ABC.SemiIndex.LinkPair;

public class CopKernelKmeans extends CopKmeans{
	Vector<LinkPair> mustlink=new Vector<LinkPair>();
	Vector<LinkPair> canotlink=new Vector<LinkPair>();	
	
	protected static KernelDistances m_DistanceFunction = new KernelDistances();
	
	
	public void setKernelpar(double par){
		m_DistanceFunction.setpar(par);
	}
	
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

	
	
}
