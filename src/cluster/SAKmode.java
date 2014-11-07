package cluster;

import java.io.File;
import java.io.FileReader;
import java.util.Random;
import java.util.Vector;

import javax.swing.JFrame;

import UFS.DissimilarityForKmodes;

import org.apache.commons.math.stat.descriptive.rank.Max;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.category.DefaultCategoryDataset;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.ui.ApplicationFrame;

import weka.clusterers.NumberOfClustersRequestable;
import weka.clusterers.RandomizableClusterer;
import weka.core.DistanceFunction;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;
import weka.gui.visualize.Plot2D;
import weka.gui.visualize.PlotData2D;

public class SAKmode extends Kmode {
	
	final int TEST_T0_COUNT=20;
	protected double tpar=0.95;
	public boolean dynamicpobfun=true;
	protected double bestfunvalue=-1;
	protected double pmutate0=0.5;
	
	void setpmutate(double p){
		pmutate0=p;
	}
	
	
	public void setDynamicpobfun(boolean flag){
		dynamicpobfun=flag;
	}
	public boolean getDynamicpobfun(boolean flag){
		return dynamicpobfun;
	}

	public double getbestvalue(){
		return this.bestfunvalue;
	}
	
	public double[] DisToProbailities(double[] dis){
		double maxdis=dis[0];
		double mindis=dis[0];
		int minsel=0;
		for(int i=1;i<dis.length;i++){
			if(dis[i]>maxdis)
				maxdis=dis[i];
			if(dis[i]<mindis){
				mindis=dis[i];
				minsel=i;
			}
		}
		double[] pob=new double[dis.length];double sumweight=0;
		for(int i=0;i<dis.length;i++){
			if(i!=minsel){
				pob[i]=maxdis/dis[i];
				sumweight+=pob[i];
			}
		}
		for(int i=0;i<dis.length;i++){
			if(i!=minsel){
				pob[i]=pob[i]/sumweight;
			}else{
				pob[i]=0;
			}
		}
		return pob;
	}
	
	public double[] Getprobailities(double[] dis,double Tpar){
		int K=dis.length;
		double[] pob=new double[K];
		double[] weight=new double[K];
		double mindis=dis[0];int minpos=0;
		double avgdis=dis[0];
		double maxdis=dis[0];
		for(int i=1;i<K;i++){
			if(dis[i]<mindis){
				mindis=dis[i];
				minpos=i;
			}
			if(dis[i]>maxdis){
				maxdis=dis[i];
			}
			avgdis+=dis[i];
		}
		avgdis=avgdis/K;
		double sumOfweight=0;
		for(int i=0;i<K;i++){
			if(i==minpos)
				weight[i]=1;
			else
				weight[i]=Math.pow(Tpar, (dis[i]/mindis));
			sumOfweight+=weight[i];
		}
		for(int i=0;i<K;i++){
			pob[i]=weight[i]/sumOfweight;
		}
		return pob;
	}
/*
	public double optfun(Instances data,Instances centres,int[] cluster){
		double fvalue=0;
		for(int i=0;i<data.numInstances();i++){
			fvalue+=m_DistanceFunction.distance(data.instance(i),centres.instance(cluster[i]));
		}
		return fvalue;
	}
*/	

	public int clusteronedata(double[] pob,double p){
		for(int i=0;i<pob.length;i++){
			if(p<pob[i])
				return i;
			p-=pob[i];
		}
		return pob.length-1;
	}
	
	
	public double evaluateT0(Instances data){
		double maxfunvalue=-1;double minfunvalue=-1;
		for(int iter=0;iter<TEST_T0_COUNT;iter++){
			Instances tempcentres=randomCentres(data,m_NumClusters,random);
			int[] tmp_clusters=new int[data.numInstances()];
			for (int i = 0; i < data.numInstances(); i++) {
				double[] dis = new double[tempcentres.numInstances()];
				for (int j = 0; j < dis.length; j++) {
					dis[j] = m_DistanceFunction.distance(data.instance(i),
							tempcentres.instance(j));
				}
				tmp_clusters[i]=Utils.minIndex(dis);
			}
			tempcentres=getNewCentres(data, tmp_clusters,
					m_NumClusters);
			double tmpf = optfun(data, tempcentres, tmp_clusters);
	        if(maxfunvalue<0 || maxfunvalue<tmpf)
	        	maxfunvalue=tmpf;
	        if(minfunvalue<0 || minfunvalue>tmpf)
	            minfunvalue=tmpf;
		}
		return (minfunvalue-maxfunvalue)/Math.log(0.8);

	}
	
	
	@Override
	public void buildClusterer(Instances data) throws Exception {
		// TODO Auto-generated method stub
		optvalueseq=new Vector<Double>();
//		Random random=new Random();
		double T0=100;
		T0=evaluateT0(data);
		if(T0<100)
			T0=100;
		//System.out.printf(T0+"\n");
		Instances centres=randomCentres(data,m_NumClusters,random);
		int[] new_clusters=new int[data.numInstances()];
		double T=T0;
		
		double pmutate=pmutate0;
		int Tpos=0;
		double curfunvalue=-1;
		bestfunvalue=-1;
		while (T > 1e-5) {
			int Titer = 0;
			while (Titer < 1) {
				Titer++;
				for (int i = 0; i < data.numInstances(); i++) {
					double[] dis = new double[centres.numInstances()];
					for (int j = 0; j < dis.length; j++) {
						dis[j] = m_DistanceFunction.distance(data.instance(i),
								centres.instance(j));
					}
					if(random.nextDouble()<pmutate){
						double[] pob = new double[dis.length];
					//	if (dynamicpobfun) {
					//		pob = Getprobailities(dis, (T / T0));
					//	} else {
						//pob = DisToProbailities(dis);
					//}
						int minsel=Utils.minIndex(dis);
						for(int pobiter=0;pobiter<pob.length;pobiter++){
							if(minsel!=pobiter)
								pob[pobiter]=1/(pob.length-1);
						}
						new_clusters[i] = clusteronedata(pob, random.nextDouble());
					}else{
						new_clusters[i]=Utils.minIndex(dis);
					}
				}
				
				Instances newcentres = getNewCentres(data, new_clusters,
						m_NumClusters);
				double newfunvalue = optfun(data, newcentres, new_clusters);
				double edown = newfunvalue - curfunvalue;
				//扰动概率充分小并且算法没有改进时。
				if( Math.pow((1-pmutate),data.numInstances())>0.995){
						if(curfunvalue>-1 && edown>=0){
							return;
						}
				}
				//if(edown>0)
					//System.out.printf(Math.exp((-edown / T))+ " "+ T + "\n");			
				if (curfunvalue < 0 || edown <= 0) {
					centres = newcentres;
					curfunvalue = newfunvalue;
					//m_clusters = new_clusters;
				}else if (random.nextDouble() < Math.exp((-edown / T))) {
					//System.out.printf(Math.exp((-edown / T))+" "+ T + -edown + "\n");
					centres = newcentres;
					curfunvalue = newfunvalue;
					//m_clusters = new_clusters;
				}
				if(bestfunvalue<0|| bestfunvalue > curfunvalue){
					bestfunvalue=curfunvalue;
					m_clusters = new_clusters;
					m_ClusterCentroids=new Instances(centres);
				}
				optvalueseq.add(curfunvalue);
			}
			Tpos++;
			T = tpar*T;
			//T = T0/(1+Tpos);
			pmutate=((T/T0)*pmutate0);
		}
	}

	public void setTpar(double p){
		tpar=p;
	}

	public static void testanddraw(Instances newdataset,int K) throws Exception{

		Kmode kmode=Kmode.RunWithDataset(newdataset, K,new Random());
		Vector<Double> kmodevalueseq=kmode.getoptseq();
		System.out.printf(kmodevalueseq.size()+ " " + kmode.getbestvalue() +"\n");
		
		XYSeries plotdatas1= new XYSeries("Kmode");
		for(int i=0;i<kmodevalueseq.size();i++){
			plotdatas1.add(i,kmodevalueseq.elementAt(i).doubleValue());
		}
		
		SAKmode sakmode=SAKmode.RunWithDataset(newdataset, K,new Random());
		Vector<Double> sakmodevalueseq=sakmode.getoptseq();
		System.out.printf(sakmodevalueseq.size()+ " " + sakmode.getbestvalue()+"\n");

		XYSeries plotdatas2 = new XYSeries("SA Kmode");
		for(int i=0;i<sakmodevalueseq.size();i++){
			plotdatas2.add(i,sakmodevalueseq.elementAt(i).doubleValue());
		}

		XYSeriesCollection  plotdataset=  new XYSeriesCollection();
		
		plotdataset.addSeries(plotdatas1);
		plotdataset.addSeries(plotdatas2);
		JFreeChart chart = ChartFactory.createScatterPlot(
	            "Optfuncvalue",                  // chart title
	            "X",                      // x axis label
	            "Y",                      // y axis label
	            plotdataset,                  // data
	            PlotOrientation.VERTICAL,
	            true,                     // include legend
	            true,                     // tooltips
	            false                     // urls
	        );
		XYPlot plot = (XYPlot) chart.getPlot();
		XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer();
        renderer.setSeriesLinesVisible(0, true);
        plot.setRenderer(renderer);
        ChartPanel chartPanel = new ChartPanel(chart);
        chartPanel.setPreferredSize(new java.awt.Dimension(500, 270));
        ApplicationFrame frame = new ApplicationFrame("Title");
        frame.setContentPane(chartPanel);
        frame.pack();
        frame.setVisible(true);
	}
	

	
	public static SAKmode RunWithDataset(Instances newdataset,int K,Random rand) throws Exception{
		SAKmode SAkmodecluster=new SAKmode();
		SAkmodecluster.setNumClusters(K);
		SAkmodecluster.setRandom(rand);
//		SAkmodecluster.setDistanceFun(new EuclideanDistance());
		SAkmodecluster.setpmutate(0.5);
		SAkmodecluster.getDistanceFun().setInstances(newdataset);
//		SAkmodecluster.setDynamicpobfun(false);
		SAkmodecluster.buildClusterer(newdataset);
		return SAkmodecluster;
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
		testanddraw(newdataset,K);
		SAKmode sakmode=SAKmode.RunWithDataset(newdataset, K,new Random());
		int[] res=sakmode.getfinalcluster();
		ClusterEvaluate.drawVat(newdataset, res);
		
		//Kmode.RunWithDataset(newdataset, K);
		SAKmode.RunWithDataset(newdataset, K,new Random());
	/*	
		int TEST_NUM=100;
		int winfac=0,losefac=0;
		double[] kmodevalues=new double[TEST_NUM];
		double[] sakmodevalues=new double[TEST_NUM];
		double sumkmodevalues=0,sumsakmodevalues=0;
		for(int i=0;i<TEST_NUM;i++){
			double curkmodevalue=Kmode.RunWithDataset(newdataset, K).getbestvalue();
			double cursakmodevalue=SAKmode.RunWithDataset(newdataset, K).getbestvalue();
			kmodevalues[i]=curkmodevalue;
			sakmodevalues[i]=cursakmodevalue;
			sumkmodevalues+=curkmodevalue;
			sumsakmodevalues+=cursakmodevalue;
			if(cursakmodevalue<curkmodevalue)
				winfac++;
			else if(curkmodevalue<cursakmodevalue){
				losefac++;
			}
		}
		Myploter.draw2valueseq(kmodevalues,sakmodevalues,"K-mode","SA-Kmode");
		double meankmodevalues=sumkmodevalues/TEST_NUM;
		double meansakmodevalues=sumsakmodevalues/TEST_NUM;
		double varkmodevalues=0,varsakmodevalues=0;
		for(int i=0;i<TEST_NUM;i++){
			varkmodevalues+=(kmodevalues[i]-meankmodevalues)*(kmodevalues[i]-meankmodevalues);
			varsakmodevalues+=(sakmodevalues[i]-meansakmodevalues)*(sakmodevalues[i]-meansakmodevalues);
		}
		varkmodevalues/=TEST_NUM;
		varsakmodevalues/=TEST_NUM;
//		Arrays.
		System.out.printf("Win: "+winfac+"\n"+"Lose:"+losefac +"\n");		
		System.out.printf("Sd for kmode: "+Math.sqrt(varkmodevalues) +"\n"+"Sd for sakmode: "+Math.sqrt(varsakmodevalues) +"\n");
*/
	}
}
