package cluster;

import java.util.Arrays;
import java.util.Vector;

import cluster.Ki.CentreswithFreq;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

public class FuzzyKmode  extends Kmode{
	
	double finalvalue = 0;	
	double[][] weight = null;
	static double  para = 1.1;

	
	public double getbestvalue() {
		return finalvalue;
	}
	
	public void computefuzzyvaluate(Instances datawithclass,double par) throws Exception{
		fXB=ClusterEvaluate.getfuzzyXB(datawithclass, m_ClusterCentroids, 
				weight, par);
		fFS=ClusterEvaluate.getfuzzyFS(datawithclass, m_ClusterCentroids, 
				weight, par);
		Instances newdata=new Instances(datawithclass);
		newdata.setClassIndex(-1);
		newdata.deleteAttributeAt(datawithclass.numAttributes()-1);
		fopt=ClusterEvaluate.getOptFun_f(newdata, m_ClusterCentroids, 
				weight, par);
	}
	
	public Instances getNewCentres(Instances data, double[][] weight, int K) {
		Instances centres = new Instances(data, K);
		for (int i = 0; i < K; i++) {
			Instance tmp = new Instance(data.numAttributes());
			for (int k = 0; k < data.numAttributes(); k++) {
				int numvalue=0;
				if(data.attribute(k).isNominal()){
					numvalue=data.attributeStats(k).distinctCount;
					numvalue=Math.max(data.attribute(k).numValues(), numvalue);
				}else{
					numvalue=(int)data.attributeStats(k).numericStats.max;
				}
				double[] count = new double[numvalue+1];//.attribute(k).numValues()
				Arrays.fill(count, 0);
				double sumcount = 0;
				for (int j = 0; j < data.numInstances(); j++) {
					int cur = (int) (1e-5 + data.instance(j).value(k));
					// System.out.printf(weight[j][i]+ " "+
					// Math.pow(weight[j][i],para)+"\n");
					count[cur] += Math.pow(weight[j][i], para);
					sumcount += Math.pow(weight[j][i], para);
				}
				int sel = Utils.maxIndex(count);
				tmp.setValue(k, sel);
			}
			centres.add(tmp);
		}
		return centres;
	}
	
	public double[][] getWeight(Instances centres, Instances data) {
		double[][] ourweight = new double[data.numInstances()][centres.numInstances()];
		double[] dis = new double[centres.numInstances()];
		for (int i = 0; i < data.numInstances(); i++) {
			int flag = -1;
			for (int j = 0; j < centres.numInstances(); j++) {
				dis[j] = m_DistanceFunction.distance(data.instance(i), centres.instance(j));
				if (dis[j] == 0) {
					flag = j;
					break;
				}
			}
			if (flag < 0) {
				if (para > 1) {
					for (int j = 0; j < centres.numInstances(); j++) {
						double sum = 0;
						for (int k = 0; k < centres.numInstances(); k++) {
							sum += Math
									.pow((dis[j] / dis[k]), 1.0 / (para - 1));
						}
						ourweight[i][j] = 1 / sum;
					}
				} else {
					int sel = Utils.minIndex(dis);
					for (int j = 0; j < centres.numInstances(); j++) {
						if (j != sel) {
							ourweight[i][j] = 0;
						} else {
							ourweight[i][j] = 1;
						}
					}
				}
			} else {
				for (int j = 0; j < centres.numInstances(); j++) {
					if (j != flag) {
						ourweight[i][j] = 0;
					} else {
						ourweight[i][j] = 1;
					}
				}
			}
		}
		return ourweight;
	}

	
	public double optfun_f(Instances data, Instances centres,
			double[][] weight) {
		double fvalue = 0;
		for (int i = 0; i < data.numInstances(); i++) {
			for (int j = 0; j < centres.numInstances(); j++) {
				double dis = m_DistanceFunction.distance(data.instance(i), centres.instance(j));
				fvalue += Math.pow(weight[i][j], para) * dis;
			}
		}
		return fvalue;
	}

	
	public void buildClusterer(Instances data) throws Exception {
		// TODO Auto-generated method stub
		int itercount = 0;
		optvalueseq = new Vector<Double>();
		m_ClusterCentroids = randomCentres(data, m_NumClusters, random);
		weight = getWeight(m_ClusterCentroids, data);
		double prefunvalue = -1;
		double optfunvalue = prefunvalue;
		for (int iter = 0; iter < maxiter; iter++) {
			itercount++;
			m_ClusterCentroids = getNewCentres(data, weight, m_NumClusters);
			weight = getWeight(m_ClusterCentroids, data);
			optfunvalue = optfun_f(data, m_ClusterCentroids, weight);
			if (prefunvalue > -1 && optfunvalue - prefunvalue > -1e-4)
				break;
			prefunvalue = optfunvalue;
		}
		m_clusters = new int[data.numInstances()];
		for (int i = 0; i < data.numInstances(); i++) {
			m_clusters[i] = Utils.maxIndex(weight[i]);
		}
		finalvalue = optfun(data, m_ClusterCentroids, m_clusters);
	}


}
