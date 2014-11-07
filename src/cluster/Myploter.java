package cluster;

import java.awt.Font;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.Vector;

import javax.swing.JFrame;

import UFS.DissimilarityForKmodes;

import org.apache.commons.math.stat.descriptive.rank.Max;
import org.apache.poi.xssf.usermodel.XSSFRow;
import org.apache.poi.xssf.usermodel.XSSFSheet;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartFrame;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.ChartUtilities;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.CategoryAxis;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.axis.ValueAxis;
import org.jfree.chart.plot.CategoryPlot;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.chart.servlet.ServletUtilities;
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


import org.jzy3d.chart.Chart;
import org.jzy3d.chart.ChartLauncher;
import org.jzy3d.colors.Color;
import org.jzy3d.colors.ColorMapper;
import org.jzy3d.colors.colormaps.ColorMapRainbow;
import org.jzy3d.maths.Coord3d;
import org.jzy3d.maths.Range;
import org.jzy3d.plot3d.builder.Builder;
import org.jzy3d.plot3d.builder.Mapper;
import org.jzy3d.plot3d.builder.concrete.OrthonormalGrid;
import org.jzy3d.plot3d.primitives.Point;
import org.jzy3d.plot3d.primitives.Scatter;
import org.jzy3d.plot3d.primitives.Shape;
import org.jzy3d.plot3d.primitives.Polygon; 

import com.mathworks.toolbox.javabuilder.MWClassID;
import com.mathworks.toolbox.javabuilder.MWException;
import com.mathworks.toolbox.javabuilder.MWNumericArray;

import showFigure.*;

public class Myploter {

	static XSSFWorkbook book;
	static XSSFSheet sheet;
	static XSSFRow row;
	
	static class myInteger implements Comparable{
		int value;
		boolean shown;
		public myInteger(int v,boolean s){
			this.value=v;
			this.shown=s;
		}
		public int compareTo(Object o1){
			if(this.value< ((myInteger)o1).value)
				return 1;
			else if(this.value > ((myInteger)o1).value)
				return -1;
			else
				return 0;
		}
		public String toString(){
			if(shown==false)
				return "";
			else
				return Integer.toString(value);
		}
	}
	
	public static JFreeChart drawfreqhist(int[] data,int min,int max,String dataname,int maxy){
		JFreeChart chart=drawfreqhist(data,min,max,dataname);
		CategoryPlot  plot = chart.getCategoryPlot();
		ValueAxis rAxis = plot.getRangeAxis(); 
		rAxis.setUpperBound(maxy);
		return chart;
	}
	
	
	public static JFreeChart drawfreqhist(int[] data,int min,int max,String dataname){
		int[] count=new int[max-min+1];
		Arrays.fill(count, 0);
		int[] x=new int[max-min+1];
		for(int i=0;i<x.length;i++){
			x[i]=min+i;
		}
		for(int i=0;i<data.length;i++){
			if(data[i]<=max && data[i]>=min)
			count[data[i]-min]++;
		}
		DefaultCategoryDataset dataset = new DefaultCategoryDataset();
		for (int i = 0; i < x.length; i++) {
			if(count[i]>0)
				dataset.addValue(count[i], "", new myInteger(x[i],true));
			else
				dataset.addValue(count[i], "", new myInteger(x[i],false));
		}		
		JFreeChart chart = ChartFactory.createBarChart3D(dataname, "", "", dataset, PlotOrientation.VERTICAL, false, false, false);
		CategoryPlot  plot = chart.getCategoryPlot();
		NumberAxis rangeAxis = (NumberAxis) plot.getRangeAxis();
		rangeAxis.setStandardTickUnits(NumberAxis.createIntegerTickUnits());
				
		plot.getDomainAxis().setLabelFont(new Font("黑体",0,10));
		plot.getDomainAxis().setTickLabelFont(new Font("黑体",0,10));
		plot.getRangeAxis().setLabelFont(new Font("黑体",0,20));
		plot.getRangeAxis().setTickLabelFont(new Font("黑体",0,20));
		plot.setBackgroundPaint(java.awt.Color.WHITE);
		
		chart.setBackgroundPaint(java.awt.Color.WHITE);
		chart.getTitle().setFont(new Font("黑体",0,30));

//		chart.getLegend(0).setItemFont(new Font("黑体",0,25));
		return chart;
	}
	
	public static Chart draw3dvalueseq(double[] x, double[] y,double[][] savalue3,String dataname,String xlab,String ylab,String zlab){
//			XYBlockRenderer= new XYBlockRenderer();
//			XYBlockRenderer.

		List<Polygon> polygons = new ArrayList<Polygon>();

		for(int i=0;i<x.length-1;i++){
			for(int j=0;j<x.length-1;j++){
				Polygon polygon = new Polygon();
				polygon.add(new Point(new Coord3d(x[i],y[j],savalue3[i][j])));
				polygon.add(new Point( new Coord3d(x[i],y[j+1],savalue3[i][j+1])));
	            polygon.add(new Point( new Coord3d(x[i+1],y[j+1],savalue3[i+1][j+1])));
	            polygon.add(new Point( new Coord3d(x[i+1],y[j],savalue3[i+1][j])));
	            polygons.add(polygon);
	            polygons.add(polygon);
			}
		}
		Shape surface = new Shape(polygons);
		surface.setColorMapper(new ColorMapper(new ColorMapRainbow(), surface.getBounds().getZmin(), surface.getBounds().getZmax(), new org.jzy3d.colors.Color(1,1,1,1f)));
	    surface.setWireframeDisplayed(true);
	    surface.setWireframeColor(org.jzy3d.colors.Color.BLACK);
	    
	    Chart chart = new Chart();
	    chart.getScene().getGraph().add(surface);
	    ChartLauncher.openChart(chart);
		return chart;
	}
	

	public static JFreeChart draw2valueseq(double[] valueseq1, double[] valueseq2,
			String s1, String s2,String dataname,String xlab,String ylab) throws Exception {
		
		XYSeries plotdatas1 = new XYSeries(s1);
		for (int i = 0; i < valueseq1.length; i++) {
			plotdatas1.add(i, valueseq1[i]);
		}
		XYSeries plotdatas2 = new XYSeries(s2);
		for (int i = 0; i < valueseq2.length; i++) {
			plotdatas2.add(i, valueseq2[i]);
		}
		XYSeriesCollection plotdataset = new XYSeriesCollection();

		plotdataset.addSeries(plotdatas1);
		plotdataset.addSeries(plotdatas2);
		JFreeChart chart = ChartFactory.createScatterPlot(dataname, // chart
																				// title
				xlab, // x axis label
				ylab, // y axis label
				plotdataset, // data
				PlotOrientation.VERTICAL, true, // include legend
				true, // tooltips
				false // urls
				);
		XYPlot plot = (XYPlot) chart.getPlot();
		XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer();
		renderer.setSeriesLinesVisible(0, true);
		plot.setRenderer(renderer);
		plot.getDomainAxis().setLabelFont(new Font("黑体",0,20));
		plot.getDomainAxis().setTickLabelFont(new Font("黑体",0,20));
		plot.getRangeAxis().setLabelFont(new Font("黑体",0,20));
		plot.getRangeAxis().setTickLabelFont(new Font("黑体",0,20));
		plot.setBackgroundPaint(java.awt.Color.WHITE);
		
		//plot.get
		chart.setBackgroundPaint(java.awt.Color.WHITE);		
		chart.getTitle().setFont(new Font("黑体",0,30));
		chart.getLegend(0).setItemFont(new Font("黑体",0,25));
//		ChartPanel chartPanel = new ChartPanel(chart);
//		chartPanel.setPreferredSize(new java.awt.Dimension(500, 270));
//		ApplicationFrame frame = new ApplicationFrame("Title");
//		frame.setContentPane(chartPanel);
//		frame.pack();
//		frame.setVisible(true);
		return chart;
	}

	
	
	public static JFreeChart draw2valueseq(double[] valueseq1, double[] valueseq2,
			String s1, String s2,String dataname,String xlab,String ylab,double[] x) throws Exception {
		XYSeries plotdatas1 = new XYSeries(s1);
		for (int i = 0; i < valueseq1.length; i++) {
			plotdatas1.add(x[i], valueseq1[i]);
		}
		XYSeries plotdatas2 = new XYSeries(s2);
		for (int i = 0; i < valueseq2.length; i++) {
			plotdatas2.add(x[i], valueseq2[i]);
		}
		XYSeriesCollection plotdataset = new XYSeriesCollection();

		plotdataset.addSeries(plotdatas1);
		plotdataset.addSeries(plotdatas2);
		JFreeChart chart = ChartFactory.createScatterPlot(dataname, // chart
																				// title
				xlab, // x axis label
				ylab, // y axis label
				plotdataset, // data
				PlotOrientation.VERTICAL, true, // include legend
				true, // tooltips
				false // urls
				);
		XYPlot plot = (XYPlot) chart.getPlot();
		XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer();
		renderer.setSeriesLinesVisible(0, true);
		plot.setRenderer(renderer);
		plot.getDomainAxis().setLabelFont(new Font("黑体",0,20));
		plot.getDomainAxis().setTickLabelFont(new Font("黑体",0,20));
		plot.getRangeAxis().setLabelFont(new Font("黑体",0,20));
		plot.getRangeAxis().setTickLabelFont(new Font("黑体",0,20));
		plot.setBackgroundPaint(java.awt.Color.WHITE);
		
		//plot.get
		chart.setBackgroundPaint(java.awt.Color.WHITE);		
		chart.getTitle().setFont(new Font("黑体",0,30));
		chart.getLegend(0).setItemFont(new Font("黑体",0,25));
//		ChartPanel chartPanel = new ChartPanel(chart);
//		chartPanel.setPreferredSize(new java.awt.Dimension(500, 270));
//		ApplicationFrame frame = new ApplicationFrame("Title");
//		frame.setContentPane(chartPanel);
//		frame.pack();
//		frame.setVisible(true);
		return chart;
	}

	public static void draw2Ddata(Instances data,int[] label,String name) throws MWException{
		
		int n = data.numInstances();
		double[][] pData = new double[n][3];
		for(int i=0;i<n;++i){
			pData[i][0]=data.instance(i).value(0);
			pData[i][1]=data.instance(i).value(1);
			pData[i][2]=label[i];
		}
		Object[] rhs = new Object[1];
		rhs[0] = new MWNumericArray(pData, MWClassID.SINGLE);
	    showFigureclass sf = new showFigureclass();
	    sf.showFigureOriginal(rhs);
	    rhs=new Object[1];
	    rhs[0]=name;
	    sf.saveFigure(rhs);
//	    sf.saveFigure(1, name);
//	    sf.saveFigure(rhs);
	}
	
	
	public static JFreeChart drawvalueseq(double[][] valueseq, String[] namestr,String dataname,String xlab,String ylab)
			throws Exception {
		if (valueseq.length != namestr.length)
			return null;
		XYSeries[] plotdata = new XYSeries[namestr.length];
		XYSeriesCollection plotdataset = new XYSeriesCollection();
		for (int i = 0; i < plotdata.length; i++) {
			plotdata[i] = new XYSeries(namestr[i]);
			for (int j = 0; j < valueseq[i].length; j++) {
				plotdata[i].add(j, valueseq[i][j]);
			}
			plotdataset.addSeries(plotdata[i]);
		}
		JFreeChart chart = ChartFactory.createScatterPlot(dataname, // chart															// title
				xlab, // x axis label
				ylab, // y axis label
				plotdataset, // data
				PlotOrientation.VERTICAL, true, // include legend
				true, // tooltips
				false // urls
				);
		XYPlot plot = (XYPlot) chart.getPlot();
		XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer();
		renderer.setSeriesLinesVisible(0, true);
		plot.setRenderer(renderer);
		plot.getDomainAxis().setLabelFont(new Font("黑体",0,20));
		plot.getDomainAxis().setTickLabelFont(new Font("黑体",0,20));
		plot.getRangeAxis().setLabelFont(new Font("黑体",0,20));
		plot.getRangeAxis().setTickLabelFont(new Font("黑体",0,20));
		plot.setBackgroundPaint(java.awt.Color.WHITE);
		
		//plot.get
		chart.setBackgroundPaint(java.awt.Color.WHITE);		
		chart.getTitle().setFont(new Font("黑体",0,30));
		chart.getLegend(0).setItemFont(new Font("黑体",0,25));
		// ChartPanel chartPanel = new ChartPanel(chart);
		// chartPanel.setPreferredSize(new java.awt.Dimension(500, 270));

		// ApplicationFrame frame = new ApplicationFrame("Title");
		// frame.setContentPane(chartPanel);
		// frame.pack();
		// frame.setVisible(true);
		return chart;
	}

	
	public static JFreeChart drawvalueseq(double[][] valueseq, String[] namestr,String dataname,String xlab,String ylab,double[] x)
			throws Exception {
		if (valueseq.length != namestr.length)
			return null;
		XYSeries[] plotdata = new XYSeries[namestr.length];
		XYSeriesCollection plotdataset = new XYSeriesCollection();
		for (int i = 0; i < plotdata.length; i++) {
			plotdata[i] = new XYSeries(namestr[i]);
			for (int j = 0; j < valueseq[i].length; j++) {
				plotdata[i].add(x[j], valueseq[i][j]);
			}
			plotdataset.addSeries(plotdata[i]);
		}
		JFreeChart chart = ChartFactory.createScatterPlot(dataname, // chart															// title
				xlab, // x axis label
				ylab, // y axis label
				plotdataset, // data
				PlotOrientation.VERTICAL, true, // include legend
				true, // tooltips
				false // urls
				);
		XYPlot plot = (XYPlot) chart.getPlot();
		XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer();
		renderer.setSeriesLinesVisible(0, true);
		plot.setRenderer(renderer);
		plot.getDomainAxis().setLabelFont(new Font("黑体",0,20));
		plot.getDomainAxis().setTickLabelFont(new Font("黑体",0,20));
		plot.getRangeAxis().setLabelFont(new Font("黑体",0,20));
		plot.getRangeAxis().setTickLabelFont(new Font("黑体",0,20));
		plot.setBackgroundPaint(java.awt.Color.WHITE);
		
		//plot.get
		chart.setBackgroundPaint(java.awt.Color.WHITE);		
		chart.getTitle().setFont(new Font("黑体",0,30));
		chart.getLegend(0).setItemFont(new Font("黑体",0,25));
		// ChartPanel chartPanel = new ChartPanel(chart);
		// chartPanel.setPreferredSize(new java.awt.Dimension(500, 270));

		// ApplicationFrame frame = new ApplicationFrame("Title");
		// frame.setContentPane(chartPanel);
		// frame.pack();
		// frame.setVisible(true);
		return chart;
	}

	
	public static void saveAsFile(JFreeChart chart, String outputPath,
			int weight, int height) {
		FileOutputStream out = null;
		try {
			File outFile = new File(outputPath);
			if (!outFile.getParentFile().exists()) {
				outFile.getParentFile().mkdirs();
			}
			out = new FileOutputStream(outputPath);
			// 保存为PNG文件
			ChartUtilities.writeChartAsJPEG(out, chart, weight, height);
			//ChartUtilities.writeChartAsPNG(out, chart, weight, height);
			// out.flush();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			if (out != null) {
				try {
					out.close();
				} catch (IOException e) {
					// do nothing
				}
			}
		}
	}

	public static void showmychart(JFreeChart chart) {
		ChartFrame cf = new ChartFrame("Tile", chart);
		cf.pack();
		cf.setVisible(true);
	}
	
	
	public static void outheader(BufferedWriter output,String dataname) throws IOException{
		output.write("\\begin{table}[htbp]\n");
		output.write("\\begin{center}\n");
		output.write("\\caption{Experimental result for " + dataname +"}\n");
		output.write("\\begin{tabular}{ c c c c c }\n");
		output.write("\\hline\n");
		output.write(" & Kmode & SA-Kmode-W & SA-Kmode-C & SA-Kmode-WC \\\\ \\hline  \n");
	}
	
	public static void outfooter(BufferedWriter output) throws IOException{
		output.write("\\hline\n");
		output.write("\\end{tabular}\n");
		output.write("\\end{center}\n");
		output.write("\\end{table}\n");		
	}
	
	public static void outline(BufferedWriter output,String measurestr,double[] vseq) throws IOException{
		DecimalFormat df=new DecimalFormat("0.000");
		DecimalFormat df_p=new DecimalFormat("0");
		output.write(measurestr + " & "+ df.format(vseq[0]) +" & " + df.format(vseq[1]) +" & " + 
		df.format(vseq[2])+ " & " + df.format(vseq[3]) + "  \\\\ \n");
	}
	

	public static void masstest(Instances newdataset, String dataname, int K,Instances dataset)
			throws Exception {

		book = new XSSFWorkbook();
		sheet = book.createSheet(dataname);
		String[] namestr = { "K-mode", "SA-Kmode-W", "SA-kmode-C",
				"SA-kmode-WC" };
		int curline = 0;
		row = sheet.createRow(curline);
		for (int i = 0; i < namestr.length; i++) {
			row.createCell(i).setCellValue(namestr[i]);
		}
		int methodnum = namestr.length;
		int TEST_NUM = 100;
		double[][][] kmodevalues = new double[1][methodnum][TEST_NUM];

		for (int i = 0; i < TEST_NUM; i++) {
			Kmode kmode = Kmode.RunWithDataset(newdataset, K,new Random(i));
			// kmode.computevaluate(dataset);
			kmodevalues[0][0][i] = kmode.getbestvalue();
			
			
			//Ki ki =Ki.RunWithDataset(newdataset, K,new Random(i) );
			// ki.computevaluate(dataset);
			//kmodevalues[0][1][i] = ki.getbestvalue();

			//int[] klable=new int[dataset.numInstances()];
			//kmodevalues[0][1][i] = k_modesImprove.kModes(dataset, K, klable,new double[3],new Random(i));

			SAKmode sakmode = SAKmode.RunWithDataset(newdataset, K,new Random(i));
			// sakmode.computevaluate(dataset);
			kmodevalues[0][1][i] = sakmode.getbestvalue();

			SAKmode2 sakmode2 = SAKmode2.RunWithDataset(newdataset, K,new Random(i));
			// sakmode2.computevaluate(dataset);
			kmodevalues[0][2][i] = sakmode2.getbestvalue();

			SAKmode3 sakmode3 = SAKmode3.RunWithDataset(newdataset, K,new Random(i));
			// sakmode3.computevaluate(dataset);
			kmodevalues[0][3][i] = sakmode3.getbestvalue();

			curline++;
			row = sheet.createRow(curline);
			for (int j = 0; j < methodnum; j++) {
				row.createCell(j).setCellValue(kmodevalues[0][j][i]);
			}
		}
		FileOutputStream fileOut;
		fileOut = new FileOutputStream("ans\\Kmode_" + dataname + ".xlsx");
		book.write(fileOut);
		fileOut.close();
		System.out.println("写入成功，运行结束！");

		JFreeChart chart = drawvalueseq(kmodevalues[0], namestr,dataname,"","J(W,Z)");
		saveAsFile(chart, "ans\\" + dataname + "_cmp_bf.jpg", 800, 600);
		// showmychart(chart);

		double[] meankmodevalues = new double[methodnum];
		double[] maxkmodevalues = new double[methodnum];
		double[] minkmodevalues = new double[methodnum];
		Arrays.fill(maxkmodevalues, 0);
		Arrays.fill(minkmodevalues, 0);
		Arrays.fill(meankmodevalues, 0);

		for (int i = 0; i < methodnum; i++) {
			for (int j = 0; j < kmodevalues[0][0].length; j++) {
				meankmodevalues[i] += kmodevalues[0][i][j];
			}
			meankmodevalues[i] /= kmodevalues[0][0].length;
			int maxindex = Utils.maxIndex(kmodevalues[0][i]);
			int minindex = Utils.minIndex(kmodevalues[0][i]);

			maxkmodevalues[i] = kmodevalues[0][i][maxindex];
			minkmodevalues[i] = kmodevalues[0][i][minindex];
		}

		double[] sdkmodevalues = new double[methodnum];
		Arrays.fill(sdkmodevalues, 0);
		for (int i = 0; i < sdkmodevalues.length; i++) {
			for (int j = 0; j < TEST_NUM; j++) {
				sdkmodevalues[i] += (kmodevalues[0][i][j] - meankmodevalues[i])
						* (kmodevalues[0][i][j] - meankmodevalues[i]);
			}
			sdkmodevalues[i] /= TEST_NUM;
			sdkmodevalues[i] = Math.sqrt(sdkmodevalues[i]);
		}

		BufferedWriter output = null;
		File outfile = new File("ans\\kmode_" + dataname + ".txt");
		outfile.createNewFile();
		output = new BufferedWriter(new FileWriter(outfile));
		outheader(output, dataname);

		outline(output, "mean", meankmodevalues);
		outline(output, "max", maxkmodevalues);
		outline(output, "min", minkmodevalues);
		outline(output, "sd", sdkmodevalues);

		outfooter(output);
		output.close();

		DecimalFormat df = new DecimalFormat("0.0000");
		// System.out.printf("Win: "+winfac+"\n"+"Lose:"+losefac +"\n");
		for (int i = 0; i < sdkmodevalues.length; i++) {
			System.out.printf("Mean for " + namestr[i] + ":"
					+ df.format(meankmodevalues[i]) + " ");
			System.out.printf("Sd for " + namestr[i] + ":"
					+ df.format(sdkmodevalues[i]) + "\n");
		}
	}

	public static void partestforsa(Instances newdataset, String dataname, int K) throws Exception {
		int iternum=10;
		double[] pm=new double[21];
		for(int i=0;i<pm.length;i++){
			pm[i]=(double)i/pm.length;
		}
		book = new XSSFWorkbook();
		String[] namestr = { "SA-Kmode-W", "SA-kmode-C",
		"SA-kmode-WC" };
		
		sheet = book.createSheet(dataname+namestr[0]);
		int curline = 0;
		row = sheet.createRow(curline);
		row.createCell(0).setCellValue("pw");
		row.createCell(1).setCellValue("value");
		double[] savalue=new double[pm.length];
		for(int i=0;i<pm.length;i++){
			curline++;
			row = sheet.createRow(curline);
			savalue[i]=0;
			for(int j=0;j<iternum;j++){
			SAKmode SAkmodecluster=new SAKmode();
			SAkmodecluster.setNumClusters(K);
			SAkmodecluster.setpmutate(pm[i]);
			SAkmodecluster.getDistanceFun().setInstances(newdataset);
			SAkmodecluster.buildClusterer(newdataset);
			savalue[i]+=SAkmodecluster.getbestvalue();
			}
			savalue[i]/=iternum;
			row.createCell(0).setCellValue(pm[i]);
			row.createCell(1).setCellValue(savalue[i]);
		}
		
		
		sheet = book.createSheet(dataname+namestr[1]);
		curline = 0;
		row = sheet.createRow(curline);
		row.createCell(0).setCellValue("pz");
		row.createCell(1).setCellValue("value");
		double[] savalue2=new double[pm.length];
		for(int i=0;i<pm.length;i++){
			curline++;
			row = sheet.createRow(curline);
			savalue2[i]=0;
			for(int j=0;j<iternum;j++){
			SAKmode2 SAkmodecluster2=new SAKmode2();
			SAkmodecluster2.setNumClusters(K);
			SAkmodecluster2.setpmutate(pm[i]);
			SAkmodecluster2.getDistanceFun().setInstances(newdataset);
			SAkmodecluster2.buildClusterer(newdataset);
			savalue2[i]+=SAkmodecluster2.getbestvalue();
			}
			savalue2[i]/=iternum;
			row.createCell(0).setCellValue(pm[i]);
			row.createCell(1).setCellValue(savalue2[i]);
		}
		JFreeChart chart =draw2valueseq(savalue,savalue2,"SA-Kmode-W","SA-Kmode-C",dataname,"p","J(W,Z)",pm);
		saveAsFile(chart, "ans\\" + dataname + "_bf.jpg", 800, 600);

		sheet = book.createSheet(dataname+namestr[2]);
		curline = 0;
		row = sheet.createRow(curline);
		row.createCell(0).setCellValue("pw");
		row.createCell(1).setCellValue("pz");
		row.createCell(2).setCellValue("value");
		double[][] savalue3 = new double[pm.length][pm.length];
		for (int i = 0; i < pm.length; i++) {
			for (int j = 0; j < pm.length; j++) {
				curline++;
				row = sheet.createRow(curline);
				savalue3[i][j]=0;
				for(int k=0;k<iternum;k++){
				SAKmode3 SAkmodecluster3 = new SAKmode3();
				SAkmodecluster3.setNumClusters(K);
				SAkmodecluster3.setpmutate(pm[i], pm[j]);
				SAkmodecluster3.getDistanceFun().setInstances(newdataset);
				SAkmodecluster3.buildClusterer(newdataset);
				savalue3[i][j]+= SAkmodecluster3.getbestvalue();
				}
				savalue3[i][j]/=iternum;
				row.createCell(0).setCellValue(pm[i]);
				row.createCell(1).setCellValue(pm[j]);
				row.createCell(2).setCellValue(savalue3[i][j]);				
			}
		}
		FileOutputStream fileOut;
		fileOut = new FileOutputStream("ans\\Kmode_par_" + dataname + ".xlsx");
		book.write(fileOut);
		fileOut.close();
		System.out.println("写入成功，运行结束！");
		
		Chart ct=draw3dvalueseq(pm,pm,savalue3,dataname,"p","p","J(W,Z)");
		ChartLauncher.screenshot(ct, "ans\\"+ dataname + "_wc_bf.png");
	}

	
	public static void getoptseqfordata(Instances newdataset, String dataname, int K,Random rand) throws Exception {
//		Kmode kmode = Kmode.RunWithDataset(newdataset, K,new Random(11));
		SAKmode sakmode = SAKmode.RunWithDataset(newdataset, K,new Random(11));
		SAKmode2 sakmode2 = SAKmode2.RunWithDataset(newdataset, K,new Random(11));
		SAKmode3 sakmode3 = SAKmode3.RunWithDataset(newdataset, K,new Random(11));
//		double[][] iterseq=new double[3][];
		String[] namestr = {"SA-Kmode-W", "SA-kmode-C","SA-kmode-WC" };
//		XYSeries plotdatas1 = new XYSeries(namestr[0]);
//		for (int i = 0; i < kmode.getoptseq().size(); i++) {
//			plotdatas1.add(i, kmode.getoptseq().elementAt(i));
//		}
		XYSeries plotdatas2 = new XYSeries(namestr[0]);
		for (int i = 0; i < sakmode.getoptseq().size(); i++) {
			plotdatas2.add(i, sakmode.getoptseq().elementAt(i));
		}
		XYSeries plotdatas3 = new XYSeries(namestr[1]);
		for (int i = 0; i < sakmode2.getoptseq().size(); i++) {
			plotdatas3.add(i, sakmode2.getoptseq().elementAt(i));
		}
		XYSeries plotdatas4 = new XYSeries(namestr[2]);
		for (int i = 0; i < sakmode3.getoptseq().size(); i++) {
			plotdatas4.add(i, sakmode3.getoptseq().elementAt(i));
		}
		XYSeriesCollection plotdataset = new XYSeriesCollection();
//		plotdataset.addSeries(plotdatas1);
		plotdataset.addSeries(plotdatas2);
		plotdataset.addSeries(plotdatas3);
		plotdataset.addSeries(plotdatas4);
		JFreeChart chart = ChartFactory.createScatterPlot(dataname, // chart
																				// title
				"step", // x axis label
				"J(W,Z)", // y axis label
				plotdataset, // data
				PlotOrientation.VERTICAL, true, // include legend
				true, // tooltips
				false // urls
				);
		XYPlot plot = (XYPlot) chart.getPlot();
		XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer();
		renderer.setSeriesLinesVisible(0, true);
		plot.setRenderer(renderer);
		plot.getDomainAxis().setLabelFont(new Font("黑体",0,20));
		plot.getDomainAxis().setTickLabelFont(new Font("黑体",0,20));
		plot.getRangeAxis().setLabelFont(new Font("黑体",0,20));
		plot.getRangeAxis().setTickLabelFont(new Font("黑体",0,20));
		plot.setBackgroundPaint(java.awt.Color.WHITE);
		
		//plot.get
		chart.setBackgroundPaint(java.awt.Color.WHITE);		
		chart.getTitle().setFont(new Font("黑体",0,30));
		chart.getLegend(0).setItemFont(new Font("黑体",0,25));

		saveAsFile(chart, "ans\\" + dataname + "_iter.jpg", 800, 600);		
	}
	
	public static void test_100() throws Exception {
		//"20_10_5_10_10_500",,,"data50_5_5_10_5_300",
		String[] datastr = {"data15_3_3_10_10_500","data50_5_5_10_5_300","zoo","vote","car",
				"lymph","soybean","tic-tac-toe", "mushroom"};

//		String[] datastr = {"mushroom"};

		
//		String[] datastr={"vote","zoo"};
		for (int dataset_iter = 0; dataset_iter < datastr.length; dataset_iter++) {
			String dataname = datastr[dataset_iter];
			
			File f = new File("E:\\CODE\\eclipse\\SVN_lib\\XattReduct\\mydata\\"
					+ dataname + ".arff");

			
			System.out.printf("Now we are running with: "+ dataname+"\n");
			
			// set data begin
			Instances dataset = new Instances(new FileReader(f));
			for (int i = 0; i < dataset.numAttributes(); i++) {
				dataset.deleteWithMissing(i);
			}
			Instances newdataset = new Instances(dataset);
			dataset.setClassIndex(dataset.numAttributes() - 1);
			newdataset.deleteAttributeAt(newdataset.numAttributes() - 1);
			int K = dataset.classAttribute().numValues();
			masstest(newdataset,dataname,K,dataset);
//			partestforsa(newdataset,dataname,K);
			getoptseqfordata(newdataset,dataname,K,new Random(10));
		}
		//masstest(datastr);
	}	
	public static void test_pic() throws Exception {
		//"20_10_5_10_10_500",,,"data50_5_5_10_5_300",
		String[] datastr = {"soybean","lymph","car","vote","zoo","data50_5_5_10_5_300","data15_3_3_10_10_500","mushroom"};
		
//		String[] datastr={"vote","zoo"};
		for (int dataset_iter = 0; dataset_iter < datastr.length; dataset_iter++) {
			String dataname = datastr[dataset_iter];
			
			File f = new File(
					"D:\\CODE\\eclipseworkspace\\SVN_lib\\XattReduct\\mydata\\"
							+ dataname + ".arff");
			
			System.out.printf("Now we are running with: "+ dataname+"\n");
			
			// set data begin
			Instances dataset = new Instances(new FileReader(f));
			for (int i = 0; i < dataset.numAttributes(); i++) {
				dataset.deleteWithMissing(i);
			}
			Instances newdataset = new Instances(dataset);
			dataset.setClassIndex(dataset.numAttributes() - 1);
			newdataset.deleteAttributeAt(newdataset.numAttributes() - 1);
			int K = dataset.classAttribute().numValues();
//			masstest(newdataset,dataname,K,dataset);
//			partestforsa(newdataset,dataname,K);
			getoptseqfordata(newdataset,dataname,K,new Random(10));
		}
		//masstest(datastr);
	}
	public static void main(String[] args) throws Exception {
		test_pic();
//		test_100();
	}
}
