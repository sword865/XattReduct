package cluster.ABC;

import java.util.Vector;

import weka.core.DistanceFunction;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;

public class SemiIndex extends ClusterIndex {
	DistanceFunction kdfun = new KernelDistances();
	double amax=0;
	double amin=0;
	double maxEuclideanDistance=-1;
	double lambda=0;

	public static class LinkPair{
		int first;
		int second;
		double distances;
		LinkPair(int a,int b,double dis){
			first=a;
			second=b;
			distances=dis;
		}
	}
	
	Jm jmindex=null;
	
	
	public void setDistanceFunction(DistanceFunction disfun){
		kdfun=disfun;
	}
	
	Vector<LinkPair> mustlink=new Vector<LinkPair>();
	Vector<LinkPair> canotlink=new Vector<LinkPair>();		
	public SemiIndex(Instances inputdata,Vector<LinkPair> inputmustlink,
			Vector<LinkPair> inputcanotlink,DistanceFunction disfun) {
		super(inputdata);
		// TODO Auto-generated constructor stub
		//jmindex=new Jm(inputdata);	
		kdfun=disfun;
		
		mustlink.clear();
		canotlink.clear();
		for(int i=0;i<inputmustlink.size();i++){
			mustlink.add(inputmustlink.elementAt(i));
		}
		for(int i=0;i<inputcanotlink.size();i++){
			canotlink.add(inputcanotlink.elementAt(i));
		}
		
		for(int i=0;i<mustlink.size();i++){
			if(mustlink.elementAt(i).distances>amax){
				amax=mustlink.elementAt(i).distances;
			}
		}
		amin=0;
		EuclideanDistance eudldisfun=new EuclideanDistance();
		eudldisfun.setInstances(dataset);
		maxEuclideanDistance=-1;
		for(int i=0;i<dataset.numInstances();i++){
			for(int j=i+1;j<dataset.numInstances();j++){
				double tmp=Math.pow(eudldisfun.distance(dataset.instance(i), dataset.instance(j)),2);
				if(tmp>maxEuclideanDistance){
					maxEuclideanDistance=tmp;
				}
			}
		}
		
		
	}

	@Override
	public double getfitnessvalue(double[] fitinput) {
		
		double[] input=new double[fitinput.length-1];
		for(int i=0;i<input.length;i++){
			input[i]=fitinput[i];
		}
		double kerpar=fitinput[fitinput.length-1];
		
		if(kdfun instanceof DistancewithPar){
			((DistancewithPar)kdfun).setpar(kerpar);
			amax=0;
			for(int i=0;i<mustlink.size();i++){
				int a1=mustlink.elementAt(i).first;
				int a2=mustlink.elementAt(i).second;
				double curdis=kdfun.distance(dataset.instance(a1),dataset.instance(a2));
				if(curdis>amax){
					amax=curdis;
				}
			}
		}
		int cluster_k=input.length/dataset.numAttributes();
		int[] idx=new int[dataset.numInstances()];
		double[] neardistances=new double[dataset.numInstances()];
		Instances centers=new Instances(dataset,cluster_k);
		int dim=dataset.numAttributes();
		for(int i=0;i<cluster_k;i++){
			Instance tmpins=new Instance(dim);
			for(int k=0;k<dim;k++){
				tmpins.setValue(k, input[i*dim+k]);
			}
			centers.add(tmpins);
		}
		for(int i=0;i<idx.length;i++){
			double mindisvalue=Double.MAX_VALUE;int sel=-1;
			for(int j=0;j<cluster_k;j++){
				double tmpdis=kdfun.distance(dataset.instance(i),centers.instance(j));
//				for(int k=0;k<dataset.numAttributes();k++){
//					tmpdis+=Math.pow(dataset.instance(i).value(k)-input[j*dataset.numAttributes()+k] , 2);
//				}
				if(tmpdis<mindisvalue){
					mindisvalue=tmpdis;
					sel=j;
				}
			}
			idx[i]=sel;
			neardistances[i]=mindisvalue;
		}
		
		double jm=0;maxEuclideanDistance=-1;
		for(int i=0;i<neardistances.length;i++){
			jm+=neardistances[i];
			if(maxEuclideanDistance<neardistances[i]){
				maxEuclideanDistance=neardistances[i];
			}
		}
				
		double dissatnum=0;
		for(int i=0;i<mustlink.size();i++){
			int a1=mustlink.elementAt(i).first;
			int a2=mustlink.elementAt(i).second;
			if(idx[a1]!=idx[a2]){
				double weight=0;double curdis=mustlink.elementAt(i).distances;
				if(curdis<0||kdfun instanceof DistancewithPar){
					curdis=kdfun.distance(dataset.instance(a1),dataset.instance(a2));
				}
				weight=Math.max(amin, amax-curdis);
				dissatnum+=weight;
			}
		}
		for(int i=0;i<canotlink.size();i++){
			int a1=canotlink.elementAt(i).first;
			int a2=canotlink.elementAt(i).second;
			if(idx[a1]==idx[a2]){
					double weight=0;double curdis=canotlink.elementAt(i).distances;
					if(curdis<0||kdfun instanceof DistancewithPar){
						curdis=kdfun.distance(dataset.instance(a1),dataset.instance(a2));
					}
					weight=Math.min(amax, amin+curdis);
					dissatnum+=weight;
			}
		}
		double unit=1.0;
		if(kdfun instanceof DistancewithPar){
			unit=maxEuclideanDistance;//((DistancewithPar)kdfun).getunitdis(maxEuclideanDistance);
		}
		double res=(jm+dissatnum);
		if(Double.isNaN(res)){
			System.out.printf("Wrong");
		}
		return  res;
		//+
		//return jmindex.getfitnessvalue(input)+dissatnum;
	}
	
}
