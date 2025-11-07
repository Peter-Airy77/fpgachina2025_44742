// ================================================================
// Module: top_enhanced
// Description: Top-level module for 16x16 pressure sensor matrix
// Data flow: Sensor -> MUX -> ADC -> FIFO -> UART -> PC
// ================================================================
module top_enhanced(
    input          sys_clk,           // System clock (50MHz)
    input          sys_rst_n,         // System reset (active low)
    // Multiplexer control interface
    output  [3:0]  row_sel,           // Row select signal
    output  [3:0]  col_sel,           // Column select signal
    // ADC interface
    input   [7:0]  ad_data_1,         // ADC data input
    input          ad_otr_1,          // ADC over-range flag
    output         ad_clk_1,          // ADC clock
    output         ad_oe_1,           // ADC output enable
    // UART interface
    output         uart_txd           // UART transmit data
    // Note: Status indicators removed - not needed for external pins
    // Can be monitored internally or through UART protocol if needed
);

// Clock and reset signals
wire        rst_n;

// Multiplexer controller signals
wire        mux_valid;
wire        frame_start;
wire        frame_done;
wire [8:0]  scan_count;
wire        adc_start;
wire        adc_ready;

// FIFO interface signals
wire        fifo_wr_en;
wire [7:0]  fifo_wr_data;
wire        fifo_rd_en;
wire [7:0]  fifo_rd_data;
wire        almost_full;
wire        almost_empty;
wire        full;
wire        empty;
wire        wr_rst_busy;
wire        rd_rst_busy;

// UART interface signals
wire        uart_tx_en;
wire        uart_tx_busy;
wire [7:0]  uart_tx_data;
wire [8:0]  read_count;

// Data control signals
wire [7:0]  ad_data_out;
wire        ad_data_valid;
wire        wr_data_valid;
wire [8:0]  data_count;

//*****************************************************
//**                    main code
//*****************************************************

// Reset signal processing
assign rst_n = sys_rst_n;

// Internal wires for status signals (not exposed to external pins)
wire        tx_busy;
wire        tx_done;
wire        frame_complete;
wire        fifo_overflow;
wire        frame_error;
wire        transmission_error;

// Multiplexer controller module
mux_controller u_mux_controller(
    .clk           (sys_clk),
    .rst_n         (rst_n),
    .row_sel       (row_sel),
    .col_sel       (col_sel),
    .mux_valid     (mux_valid),
    .scan_count    (scan_count),
    .frame_start   (frame_start),
    .frame_done    (frame_done),
    .adc_ready     (adc_ready),
    .adc_start     (adc_start)
);

// ADC controller module
hs_dual_ad u_hs_dual_ad(
    .hs_clk        (sys_clk),
    .sys_rst_n     (rst_n),
    .mux_valid     (mux_valid),
    .adc_start     (adc_start),
    .ad_data_1     (ad_data_1),
    .ad_otr_1      (ad_otr_1),
    .ad_clk_1      (ad_clk_1),
    .ad_oe_1       (ad_oe_1),
    .ad_data_out   (ad_data_out),
    .ad_data_valid (ad_data_valid),
    .adc_ready     (adc_ready),
    .ad_error      ()               // Optional error handling
);

// Enhanced FIFO write controller module
fifo_wr_enhanced u_fifo_wr_enhanced(
    .wr_clk        (sys_clk),
    .rst_n         (rst_n),
    .frame_start   (frame_start),
    .frame_done    (frame_done),
    .scan_count    (scan_count),
    .ad_data       (ad_data_out),
    .ad_data_valid (ad_data_valid),
    .ad_otr_1      (ad_otr_1),
    .wr_rst_busy   (wr_rst_busy),
    .empty         (empty),
    .almost_full   (almost_full),
    .full          (full),
    .fifo_wr_en    (fifo_wr_en),
    .fifo_wr_data  (fifo_wr_data),
    .data_valid    (wr_data_valid),
    .fifo_overflow (fifo_overflow),
    .data_count    (data_count),
    .frame_error   (frame_error)
);

// FIFO IP core instantiation
// Note: You need to generate this using Xilinx Vivado IP Catalog
// Configuration: 256 depth, 8-bit width, First Word Fall Through
fifo_generator_0 u_fifo(
    .rst           (~rst_n),
    .wr_clk        (sys_clk),
    .rd_clk        (sys_clk),
    .din           (fifo_wr_data),
    .wr_en         (fifo_wr_en),
    .rd_en         (fifo_rd_en),
    .dout          (fifo_rd_data),
    .full          (full),
    .almost_full   (almost_full),
    .empty         (empty),
    .almost_empty  (almost_empty),
    .wr_rst_busy   (wr_rst_busy),
    .rd_rst_busy   (rd_rst_busy)
);

// FIFO read controller module
fifo_rd u_fifo_rd(
    .rd_clk            (sys_clk),
    .rst_n             (rst_n),
    .rd_rst_busy       (rd_rst_busy),
    .fifo_rd_data      (fifo_rd_data),
    .full              (full),
    .almost_empty      (almost_empty),
    .empty             (empty),
    .uart_tx_busy      (uart_tx_busy),
    .fifo_rd_en        (fifo_rd_en),
    .uart_tx_en        (uart_tx_en),
    .uart_tx_data      (uart_tx_data),
    .frame_complete    (frame_complete),
    .frame_done        (frame_done),
    .read_count        (read_count),
    .transmission_error(transmission_error)
);

// UART transmitter module
uart_tx u_uart_tx(
    .hs_clk        (sys_clk),
    .rst_n         (rst_n),
    .uart_tx_en    (uart_tx_en),
    .uart_tx_data  (uart_tx_data),
    .uart_txd      (uart_txd),
    .uart_tx_busy  (uart_tx_busy),
    .tx_done       (tx_done),
    .tx_error      ()               // Optional error handling
);

endmodule

