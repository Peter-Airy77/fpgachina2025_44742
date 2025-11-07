// ================================================================
// Testbench: tb_top_enhanced
// Description: Comprehensive testbench for 16x16 pressure sensor system
// Tests: ADC data acquisition -> FIFO -> UART transmission
// Simulation tool: Modelsim + Vivado
// ================================================================
`timescale 1ns/1ps

module tb_top_enhanced();

// Parameters
parameter CLK_PERIOD = 20;        // 50MHz clock (20ns period)
parameter UART_BPS = 115200;      // UART baud rate
parameter CLK_FREQ = 50000000;    // 50MHz
parameter BIT_PERIOD = CLK_FREQ / UART_BPS * CLK_PERIOD; // UART bit period in ns

// Testbench signals
reg         sys_clk;
reg         sys_rst_n;
reg  [7:0]  ad_data_1;
reg         ad_otr_1;
wire [3:0]  row_sel;
wire [3:0]  col_sel;
wire        ad_clk_1;
wire        ad_oe_1;
wire        uart_txd;

// Test control variables
integer     i;
integer     frame_count;
integer     uart_byte_count;
reg  [7:0]  received_data [0:255];  // Array to store received UART data
reg  [7:0]  sent_data [0:255];      // Array to store sent ADC data
integer     error_count;

// UART receiver variables
reg  [7:0]  uart_rx_data;
reg         uart_rx_valid;
integer     bit_counter;
reg  [3:0]  uart_state;

// FIFO status monitoring signals (connected to DUT internal signals)
// These are for waveform observation only
wire        fifo_full;
wire        fifo_almost_full;
wire        fifo_empty;
wire        fifo_almost_empty;
wire        fifo_wr_en;
wire [7:0]  fifo_wr_data;
wire        fifo_rd_en;
wire [7:0]  fifo_rd_data;
wire        fifo_wr_rst_busy;
wire        fifo_rd_rst_busy;
wire [8:0]  fifo_data_count;

// Connect monitoring signals to DUT internal signals for waveform viewing
assign fifo_full = u_top_enhanced.full;
assign fifo_almost_full = u_top_enhanced.almost_full;
assign fifo_empty = u_top_enhanced.empty;
assign fifo_almost_empty = u_top_enhanced.almost_empty;
assign fifo_wr_en = u_top_enhanced.fifo_wr_en;
assign fifo_wr_data = u_top_enhanced.fifo_wr_data;
assign fifo_rd_en = u_top_enhanced.fifo_rd_en;
assign fifo_rd_data = u_top_enhanced.fifo_rd_data;
assign fifo_wr_rst_busy = u_top_enhanced.wr_rst_busy;
assign fifo_rd_rst_busy = u_top_enhanced.rd_rst_busy;
assign fifo_data_count = u_top_enhanced.data_count;

// Note: FIFO IP core (fifo_generator_0) simulation model is automatically
// compiled by Vivado from the IP core generation files.
// The DUT instantiates fifo_generator_0 which will use the correct simulation model.

//*****************************************************
//**                Clock Generation
//*****************************************************
initial begin
    sys_clk = 1'b0;
    forever #(CLK_PERIOD/2) sys_clk = ~sys_clk;
end

//*****************************************************
//**                Reset Generation
//*****************************************************
initial begin
    sys_rst_n = 1'b0;
    #(CLK_PERIOD * 10);
    sys_rst_n = 1'b1;
    $display("========================================");
    $display("  System Reset Released at %0t ns", $time);
    $display("========================================");
end

//*****************************************************
//**                ADC Data Simulation
//*****************************************************
// Simulate ADC providing data based on row/col selection
always @(posedge sys_clk or negedge sys_rst_n) begin
    if(!sys_rst_n) begin
        ad_data_1 <= 8'h00;
        ad_otr_1 <= 1'b0;
    end
    else begin
        // Generate test pattern: row*16 + col
        ad_data_1 <= {row_sel, col_sel};
        ad_otr_1 <= 1'b0;  // No over-range
    end
end

//*****************************************************
//**                DUT Instantiation
//*****************************************************
top_enhanced u_top_enhanced(
    .sys_clk       (sys_clk),
    .sys_rst_n     (sys_rst_n),
    .row_sel       (row_sel),
    .col_sel       (col_sel),
    .ad_data_1     (ad_data_1),
    .ad_otr_1      (ad_otr_1),
    .ad_clk_1      (ad_clk_1),
    .ad_oe_1       (ad_oe_1),
    .uart_txd      (uart_txd)
);

//*****************************************************
//**          FIFO Status Display
//*****************************************************
// Monitor FIFO status for debugging
always @(posedge sys_clk) begin
    if(fifo_wr_en) begin
        $display("[FIFO WR] Time=%0t, Data=0x%02h, Count=%0d, Full=%b, Almost_Full=%b", 
                 $time, fifo_wr_data, fifo_data_count, fifo_full, fifo_almost_full);
    end
    if(fifo_rd_en) begin
        $display("[FIFO RD] Time=%0t, Data=0x%02h, Empty=%b, Almost_Empty=%b", 
                 $time, fifo_rd_data, fifo_empty, fifo_almost_empty);
    end
end

//*****************************************************
//**                UART Receiver
//*****************************************************
// Simple UART receiver to capture transmitted data
parameter UART_IDLE  = 4'd0;
parameter UART_START = 4'd1;
parameter UART_DATA  = 4'd2;
parameter UART_STOP  = 4'd3;

initial begin
    uart_state = UART_IDLE;
    uart_rx_data = 8'h00;
    uart_rx_valid = 1'b0;
    bit_counter = 0;
    uart_byte_count = 0;
end

// UART receiver state machine
always @(negedge sys_clk) begin
    case(uart_state)
        UART_IDLE: begin
            uart_rx_valid <= 1'b0;
            if(uart_txd == 1'b0) begin  // Start bit detected
                uart_state <= UART_START;
                bit_counter <= 0;
                #(BIT_PERIOD/2);  // Sample in middle of bit
            end
        end
        
        UART_START: begin
            #BIT_PERIOD;
            uart_state <= UART_DATA;
            bit_counter <= 0;
        end
        
        UART_DATA: begin
            uart_rx_data[bit_counter] <= uart_txd;
            bit_counter <= bit_counter + 1;
            #BIT_PERIOD;
            if(bit_counter == 7) begin
                uart_state <= UART_STOP;
            end
        end
        
        UART_STOP: begin
            if(uart_txd == 1'b1) begin  // Valid stop bit
                received_data[uart_byte_count] <= uart_rx_data;
                uart_rx_valid <= 1'b1;
                $display("[UART RX] Byte %03d: 0x%02h (Row=%d, Col=%d) at %0t ns", 
                         uart_byte_count, uart_rx_data, 
                         uart_rx_data[7:4], uart_rx_data[3:0], $time);
                uart_byte_count <= uart_byte_count + 1;
            end
            else begin
                $display("[UART ERROR] Invalid stop bit at %0t ns", $time);
                error_count = error_count + 1;
            end
            #BIT_PERIOD;
            uart_state <= UART_IDLE;
        end
        
        default: uart_state <= UART_IDLE;
    endcase
end

//*****************************************************
//**                Test Monitoring
//*****************************************************
// Monitor row and column scanning
always @(posedge sys_clk) begin
    if(u_top_enhanced.u_mux_controller.current_state == 3'b001) begin  // SET_ROW_COL state
        $display("[MUX] Scanning Row=%02d, Col=%02d, Count=%03d at %0t ns", 
                 row_sel, col_sel, 
                 u_top_enhanced.u_mux_controller.scan_count, $time);
    end
end

// Monitor frame completion
always @(posedge sys_clk) begin
    if(u_top_enhanced.frame_start) begin
        $display("========================================");
        $display("[FRAME] Frame Started at %0t ns", $time);
        $display("========================================");
    end
    if(u_top_enhanced.frame_done) begin
        frame_count = frame_count + 1;
        $display("========================================");
        $display("[FRAME] Frame %0d Completed at %0t ns", frame_count, $time);
        $display("========================================");
    end
end

// Monitor FIFO status
always @(posedge sys_clk) begin
    if(u_top_enhanced.fifo_wr_en) begin
        $display("[FIFO WR] Data=0x%02h, Count=%03d at %0t ns", 
                 u_top_enhanced.fifo_wr_data, 
                 u_top_enhanced.data_count, $time);
    end
    if(u_top_enhanced.full) begin
        $display("[FIFO] FIFO Full at %0t ns", $time);
    end
end

// Monitor errors
always @(posedge sys_clk) begin
    if(u_top_enhanced.fifo_overflow) begin
        $display("[ERROR] FIFO Overflow at %0t ns", $time);
        error_count = error_count + 1;
    end
    if(u_top_enhanced.frame_error) begin
        $display("[ERROR] Frame Error at %0t ns", $time);
        error_count = error_count + 1;
    end
    if(u_top_enhanced.transmission_error) begin
        $display("[ERROR] Transmission Error at %0t ns", $time);
        error_count = error_count + 1;
    end
end

//*****************************************************
//**                Main Test Sequence
//*****************************************************
initial begin
    // Initialize variables
    frame_count = 0;
    uart_byte_count = 0;
    error_count = 0;
    
    // Initialize test data array
    for(i = 0; i < 256; i = i + 1) begin
        sent_data[i] = i;
        received_data[i] = 8'hFF;
    end
    
    // Wait for reset
    wait(sys_rst_n == 1'b1);
    #(CLK_PERIOD * 20);
    
    $display("\n");
    $display("========================================");
    $display("  Starting Functional Test");
    $display("  Testing 16x16 Sensor Matrix System");
    $display("========================================");
    $display("\n");
    
    // Wait for first frame to complete (256 points scanned)
    wait(frame_count >= 1);
    #(CLK_PERIOD * 100);
    
    // Wait for UART transmission to complete (256 bytes)
    $display("\n[TEST] Waiting for UART transmission...\n");
    wait(uart_byte_count >= 256);
    #(CLK_PERIOD * 1000);
    
    // Verify received data
    $display("\n========================================");
    $display("  Data Verification");
    $display("========================================");
    
    error_count = 0;
    for(i = 0; i < 256; i = i + 1) begin
        if(received_data[i] !== i[7:0]) begin
            $display("[VERIFY ERROR] Byte %03d: Expected=0x%02h, Received=0x%02h", 
                     i, i[7:0], received_data[i]);
            error_count = error_count + 1;
        end
    end
    
    // Test summary
    $display("\n========================================");
    $display("  Test Summary");
    $display("========================================");
    $display("  Total Frames:        %0d", frame_count);
    $display("  UART Bytes Received: %0d", uart_byte_count);
    $display("  Errors Detected:     %0d", error_count);
    
    if(error_count == 0 && uart_byte_count == 256) begin
        $display("\n  *** TEST PASSED ***");
    end
    else begin
        $display("\n  *** TEST FAILED ***");
    end
    $display("========================================\n");
    
    // Run for a bit more
    #(CLK_PERIOD * 1000);
    
    $display("Simulation completed at %0t ns", $time);
    $finish;
end

//*****************************************************
//**                Timeout Watchdog
//*****************************************************
initial begin
    #(CLK_PERIOD * 5000000);  // 100ms timeout
    $display("\n========================================");
    $display("  ERROR: Simulation Timeout!");
    $display("========================================\n");
    $finish;
end

//*****************************************************
//**                Waveform Dump (Optional)
//*****************************************************
initial begin
    $dumpfile("tb_top_enhanced.vcd");
    $dumpvars(0, tb_top_enhanced);
end

endmodule

