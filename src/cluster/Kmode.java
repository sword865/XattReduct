package cluster;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;
import java.util.Vector;

import javax.swing.JFrame;

import org.jfree.chart.ChartUtilities;
import org.jfree.chart.JFreeChart;

import UFS.DissimilarityForKmodes;


import weka.clusterers.NumberOfClustersRequestable;
import weka.clusterers.RandomizableClusterer;
import weka.core.DistanceFunction;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.WeightedInstancesHandler;

public class Kmode extends RandomizableClusterer  implements NumberOfClustersRequestable,WeightedInstancesHandler{

	protected int m_NumClusters = 2;
	protected Instances m_ClusterCentroids;
	protected static DistanceFunction m_DistanceFunction = new DissimilarityForKmodes();
	
	protected int maxiter=500;
	protected Vector<Double> optvalueseq;
	protected int[] m_clusters;	
	//protected boolean evaluate=true;
	protected double XB;
	protected double Ik;
	protected double CA;
	protected double NMI;
	protected int ACnum;
//	protected double accuray;
	
	protected double fXB;
	protected double fFS;
	protected double fopt;

	
	Random random=new Random();

	
	public double getXB(){
		return XB;
	}
	public double getIk(){
		return Ik;
	}
	public double getCA(){
		return CA;
	}
	public double getfXB(){
		return fXB;
	}
	public double getfFS(){
		return fFS;
	}
	public double getfopt(){
		return fopt;
	}
	public int getACnumber(){
		return ACnum;
	}
	public double getNMI(){
		return NMI;
	}
	public double getbestvalue(){
		return this.optvalueseq.lastElement();
	}
	
	public int[] getfinalcluster(){
		return m_clusters;
	}
	
	public void setDistanceFun(DistanceFunction disfun){
		m_DistanceFunction=disfun;
	}
	public DistanceFunction getDistanceFun(){
		return m_DistanceFunction;
	}
	
	public Vector<Double> getoptseq(){
		return optvalueseq;
	}
	
	public void setRandom(Random r){
		this.random=r;
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

	public double optfun(Instances data,Instances centres,int[] cluster){
		double fvalue=0;
		for(int i=0;i<data.numInstances();i++){
			fvalue+=m_DistanceFunction.distance(data.instance(i),centres.instance(cluster[i]));
					//*m_DistanceFunction.distance(data.instance(i),centres.instance(cluster[i]));
		}
		return fvalue;
	}

//	public void 
	public void computevaluate(Instances datawithclass) throws Exception{
		XB=ClusterEvaluate.getXB(datawithclass, m_ClusterCentroids, m_clusters);
		Ik=ClusterEvaluate.getIk(datawithclass, m_ClusterCentroids, m_clusters);
		//CA=ClusterEvaluate.getCA(datawithclass, m_clusters);
		ACnum=ClusterEvaluate.getACnumber(datawithclass, m_clusters, m_NumClusters);
		NMI=ClusterEvaluate.getNMI(datawithclass, m_clusters);
	}
	
	public void computefuzzyvaluate(Instances datawithclass,double par) throws Exception{
		fXB=ClusterEvaluate.getfuzzyXB(datawithclass, m_ClusterCentroids, 
				ClusterEvaluate.clustertoweight(m_clusters,m_NumClusters), par);
		fFS=ClusterEvaluate.getfuzzyFS(datawithclass, m_ClusterCentroids, 
				ClusterEvaluate.clustertoweight(m_clusters,m_NumClusters), par);
		Instances newdata=new Instances(datawithclass);
		newdata.setClassIndex(-1);
		newdata.deleteAttributeAt(datawithclass.numAttributes()-1);
		fopt=ClusterEvaluate.getOptFun_f(newdata, m_ClusterCentroids, 
				ClusterEvaluate.clustertoweight(m_clusters,m_NumClusters), par);
	}
	
	@Override
	public void buildClusterer(Instances data) throws Exception {
		// TODO Auto-generated method stub
		int itercount=0;
		optvalueseq=new Vector<Double>();
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
				double mindis=dis[0];int sel=0;
				for(int j=1;j<dis.length;j++){
					if(dis[j]<mindis){
						sel=j;
						mindis=dis[j];
					}
				}
				m_clusters[i]=sel;
			}
			centres=getNewCentres(data,m_clusters,m_NumClusters);
			optfunvalue=optfun(data,centres,m_clusters);
			optvalueseq.add(optfunvalue);			
			if(prefunvalue>-1 && optfunvalue-prefunvalue > -1e-8 )
				break;
			prefunvalue=optfunvalue;
		}
		m_ClusterCentroids=new Instances(centres);
		//System.out.print(itercount);
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
	
	public static Kmode RunWithDataset(Instances dataset,int K,Random rand) throws Exception{
		Kmode kmodecluster=new Kmode();
		kmodecluster.setNumClusters(K);
		kmodecluster.setRandom(rand);
//		kmodecluster.setDistanceFun(new EuclideanDistance());
		kmodecluster.getDistanceFun().setInstances(dataset);
		kmodecluster.buildClusterer(dataset);
		return kmodecluster;
	}
	

	public double getXB(Instances dataset){
		return XB;
	}
	public double getIk(Instances dataset){
		return Ik;
	}
	public double getCA(Instances dataset){
		return CA;
	}

	public static void main(String[] args) throws Exception {
		File f = new File(
				"D:\\CODE\\eclipseworkspace\\SVN_lib\\XattReduct\\mydata\\zoo.arff");
		// set data begin
		Instances dataset = new Instances(new FileReader(f));
		for(int i=0;i<dataset.numAttributes();i++){
			dataset.deleteWithMissing(i);
		}
		Instances newdataset=new Instances(dataset);
		dataset.setClassIndex(dataset.numAttributes() - 1);
		newdataset.deleteAttributeAt(newdataset.numAttributes()-1);
		int K=dataset.classAttribute().numValues();
		//System.out.printf(RunWithDataset(newdataset,K) + "\n");

		Kmode kmodecluster=new Kmode();
		kmodecluster.setNumClusters(K);
		kmodecluster.getDistanceFun().setInstances(newdataset);
		kmodecluster.buildClusterer(newdataset);
		int[] res=kmodecluster.getfinalcluster();
		ClusterEvaluate.drawVat(dataset, res);
		System.out.printf("class num:" + dataset.classAttribute().numValues());
		
	}


}
