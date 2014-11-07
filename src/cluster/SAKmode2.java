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

public class SAKmode2 extends SAKmode {
	
	
	public void setDynamicpobfun(boolean flag){
		dynamicpobfun=flag;
	}
	public boolean getDynamicpobfun(boolean flag){
		return dynamicpobfun;
	}

	public double getbestvalue(){
		return this.bestfunvalue;
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
					new_clusters[i] = Utils.minIndex(dis);
				}
				
				Instances newcentres = getNewCentres(data, new_clusters,
						m_NumClusters);
				//扰动中心点
				for(int i=0;i<newcentres.numInstances();i++){
					for (int j = 0; j < newcentres.numAttributes(); j++) {
						if (random.nextDouble() < pmutate) {
							int b = random.nextInt(data.numInstances());
							newcentres.instance(i).setValue(j,data.instance(b).value(j));
						}
					}
				}
				double newfunvalue = optfun(data, newcentres, new_clusters);
				double edown = newfunvalue - curfunvalue;
				//扰动概率充分小并且算法没有改进时。
				if( Math.pow((1-pmutate),m_NumClusters*newcentres.numAttributes())>0.995){
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
		
		SAKmode2 sakmode=SAKmode2.RunWithDataset(newdataset, K,new Random());
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
	
	public static SAKmode2 RunWithDataset(Instances newdataset,int K,Random rand) throws Exception{
		SAKmode2 SAkmode2cluster=new SAKmode2();
		SAkmode2cluster.setNumClusters(K);
		SAkmode2cluster.setRandom(rand);
//		SAkmode2cluster.setDistanceFun(new EuclideanDistance());
		SAkmode2cluster.setpmutate(0.5);
		SAkmode2cluster.getDistanceFun().setInstances(newdataset);
//		SAkmode2cluster.setDynamicpobfun(false);
		SAkmode2cluster.buildClusterer(newdataset);
		return SAkmode2cluster;
	}
	
	public static void main(String[] args) throws Exception {
		File f = new File(
				"D:\\CODE\\eclipseworkspace\\SVN_lib\\XattReduct\\mydata\\soybean.arff");
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
		//Kmode.RunWithDataset(newdataset, K);
		//SAKmode.RunWithDataset(newdataset, K);
		int TEST_NUM=100;
		int winfac=0,losefac=0;
		double[] kmodevalues=new double[TEST_NUM];
		double[] sakmodevalues=new double[TEST_NUM];
		double sumkmodevalues=0,sumsakmodevalues=0;
		for(int i=0;i<TEST_NUM;i++){
			double curkmodevalue=Kmode.RunWithDataset(newdataset, K,new Random()).getbestvalue();
			double cursakmodevalue=SAKmode2.RunWithDataset(newdataset, K,new Random()).getbestvalue();
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
		Myploter.draw2valueseq(kmodevalues,sakmodevalues,"K-mode","SA-Kmode","soybean","X","Y");
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
	}
}
