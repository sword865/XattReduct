package cluster;

import java.io.File;
import java.io.FileReader;
import java.util.Vector;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.ui.ApplicationFrame;

import weka.core.Debug.Random;
import weka.core.Instance;
import weka.core.Instances;

public class Ki extends Kmode {
	
	protected Vector<Double> optvalueseq_iter;

	
	
	class CentreswithFreq{
		public Instances centres;
		public double[][] freq;
	}
	
	public CentreswithFreq getNewCentres_KI(Instances data,int clusters[],int K){
		CentreswithFreq cwf=new CentreswithFreq();
		double[][] freq=new double[K][data.numAttributes()];
		for(int i=0;i<freq.length;i++)
			for(int j=0;j<freq[0].length;j++)
				freq[i][j]=0;
		
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
				for(int k=0;k<memberSet[i].numInstances();k++){
					if(memberSet[i].instance(k).value(j)==tmp.value(j))
						freq[i][j]++;
				}
				if(memberSet[i].numInstances()>0){
					freq[i][j]/=memberSet[i].numInstances();
				}else{
					freq[i][j]=1;
				}
			}
			centres.add(tmp);
		}
		cwf.centres=centres;
		cwf.freq=freq;
		return cwf;
	}

	public double optfun_ki(Instances data,CentreswithFreq cwf,int[] cluster){
		double fvalue=0;
		for(int i=0;i<data.numInstances();i++){			
			double dis=0;
			for(int k=0;k<data.numAttributes();k++){
				if(data.instance(i).value(k)!=cwf.centres.instance(cluster[i]).value(k)){
					dis+=1;
				}else{
					dis+=1-cwf.freq[cluster[i]][k];
				}
			}
			fvalue+=dis;
		}
		return fvalue;
	}
		
	@Override
	public void buildClusterer(Instances data) throws Exception {
		// TODO Auto-generated method stub
		int itercount=0;
		optvalueseq=new Vector<Double>();
		optvalueseq_iter=new Vector<Double>();
		CentreswithFreq cwf=new CentreswithFreq();
		cwf.centres=randomCentres(data,m_NumClusters,random);
		m_clusters=new int[data.numInstances()];
		for(int i=0;i<data.numInstances();i++){
			double[] dis=new double[cwf.centres.numInstances()];
			for(int j=0;j<dis.length;j++){
				dis[j]=m_DistanceFunction.distance(data.instance(i),cwf.centres.instance(j));
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
		double prefunvalue=-1;
		double optfunvalue=prefunvalue;		
		for(int iter=0;iter<maxiter;iter++){			
			itercount++;
			cwf=getNewCentres_KI(data,m_clusters,m_NumClusters);
			for(int i=0;i<data.numInstances();i++){
				double[] dis=new double[cwf.centres.numInstances()];
				for(int j=0;j<dis.length;j++){
					dis[j]=0;
					for(int k=0;k<data.numAttributes();k++)
						if(data.instance(i).value(k)!=cwf.centres.instance(j).value(k)){
							dis[j]+=1;
						}else{
							dis[j]+=1-cwf.freq[j][k];
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
			optfunvalue=optfun_ki(data,cwf,m_clusters);
			optvalueseq.add(optfun(data,cwf.centres,m_clusters));
			optvalueseq_iter.add(optfunvalue);			
			if(prefunvalue>-1 && optfunvalue-prefunvalue > -1e-8 )
				break;
			prefunvalue=optfunvalue;
		}
		m_ClusterCentroids=new Instances(cwf.centres);
		//System.out.print(itercount);
	}

	public static Ki RunWithDataset(Instances dataset,int K,java.util.Random rand) throws Exception{
		Ki kicluster=new Ki();
		kicluster.setNumClusters(K);
		kicluster.setRandom(rand);
//		kmodecluster.setRandom(new Random(0));
//		kmodecluster.setDistanceFun(new EuclideanDistance());
		kicluster.getDistanceFun().setInstances(dataset);
		kicluster.buildClusterer(dataset);
		return kicluster;
	}
	
	
	public static JFreeChart testanddraw(Instances newdataset,int K) throws Exception{

		Kmode kmode=Kmode.RunWithDataset(newdataset, K,new Random());
		Vector<Double> kmodevalueseq=kmode.getoptseq();
		System.out.printf(kmodevalueseq.size()+ " " + kmode.getbestvalue() +"\n");
		
		XYSeries plotdatas1= new XYSeries("Kmode");
		for(int i=0;i<kmodevalueseq.size();i++){
			plotdatas1.add(i,kmodevalueseq.elementAt(i).doubleValue());
		}
		
		Ki ki=Ki.RunWithDataset(newdataset, K,new Random());
		Vector<Double> sakmodevalueseq=ki.getoptseq();
		System.out.printf(sakmodevalueseq.size()+ " " + ki.getbestvalue()+"\n");

		XYSeries plotdatas2 = new XYSeries("Ki");
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
        return chart;
	}
	
	
	public static void main(String[] args) throws Exception {
		File f = new File(
				"D:\\CODE\\eclipseworkspace\\SVN_lib\\XattReduct\\mydata\\soybean.arff");
		// set data begin
		Instances dataset = new Instances(new FileReader(f));
		for(int i=0;i<dataset.numAttributes();i++){
		//	dataset.deleteWithMissing(i);
		}
		Instances newdataset=new Instances(dataset);
		dataset.setClassIndex(dataset.numAttributes() - 1);
		newdataset.deleteAttributeAt(newdataset.numAttributes()-1);
		int K=dataset.classAttribute().numValues();
		//JFreeChart chart=testanddraw(newdataset,K);
		//Myploter.saveAsFile(chart, "D:\\ans.png", 500 , 500);
		int TEST_NUM=100;
		int winfac=0,losefac=0;
		for(int i=0;i<TEST_NUM;i++){
			double curkmodevalue=Kmode.RunWithDataset(newdataset, K,new Random()).getbestvalue();
			double curkivalue=Ki.RunWithDataset(newdataset, K,new Random()).getbestvalue();
			if(curkivalue<curkmodevalue)
				winfac++;
			else if(curkmodevalue<curkivalue){
				losefac++;
			}
		}
		System.out.printf("Win: "+winfac+"\n"+"Lose:"+losefac +"\n");
	}
}
