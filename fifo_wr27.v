// ================================================================
// Module: fifo_wr_enhanced
// Description: Enhanced FIFO write controller with frame synchronization
// ================================================================
module fifo_wr_enhanced(
    // Clock and Reset
    input               wr_clk,           // Write clock signal
    input               rst_n,            // Active low reset
    // Frame Control Interface
    input               frame_start,      // Frame start signal
    input               frame_done,       // Frame done signal
    input        [8:0]  scan_count,       // Scan count (0-255 requires 9 bits)
    // ADC Data Interface
    input        [7:0]  ad_data,          // ADC output data
    input               ad_data_valid,    // ADC data valid
    input               ad_otr_1,         // ADC over-range flag
    // FIFO Interface
    input               wr_rst_busy,      // Write reset busy
    input               empty,            // FIFO empty
    input               almost_full,      // FIFO almost full
    input               full,             // FIFO full
    output reg          fifo_wr_en,       // FIFO write enable
    output reg   [7:0]  fifo_wr_data,     // Data to write to FIFO
    output reg          data_valid,       // Data valid indicator
    output reg          fifo_overflow,    // FIFO overflow indicator
    output reg   [8:0]  data_count,       // Data counter (0-256)
    output reg          frame_error       // Frame error indicator
);

// State machine parameters
parameter [1:0] WR_IDLE      = 2'b00;     // Idle state
parameter [1:0] WR_WAIT_DATA = 2'b01;     // Wait for valid ADC data
parameter [1:0] WR_CHECK     = 2'b10;     // Check FIFO status
parameter [1:0] WR_DATA      = 2'b11;     // Execute FIFO write

// Register declarations
reg [1:0] current_state;
reg [1:0] next_state;
reg       empty_d0;
reg       empty_d1;
reg [7:0] ad_data_sync;         // ADC data synchronization
reg       data_valid_d0;
reg       frame_active;

//*****************************************************
//**                    main code
//*****************************************************

// Synchronize empty signal to write clock domain
always @(posedge wr_clk or negedge rst_n) begin
    if(!rst_n) begin
        empty_d0 <= 1'b0;
        empty_d1 <= 1'b0;
    end
    else begin
        empty_d0 <= empty;
        empty_d1 <= empty_d0;
    end
end

// Synchronize ADC data
always @(posedge wr_clk or negedge rst_n) begin
    if(!rst_n) begin
        ad_data_sync <= 8'b0;
        data_valid_d0 <= 1'b0;
    end
    else begin
        ad_data_sync <= ad_data;
        data_valid_d0 <= ad_data_valid;
    end
end

// Frame active flag
always @(posedge wr_clk or negedge rst_n) begin
    if(!rst_n)
        frame_active <= 1'b0;
    else if(frame_start)
        frame_active <= 1'b1;
    else if(frame_done)
        frame_active <= 1'b0;
end

// Data counter logic
always @(posedge wr_clk or negedge rst_n) begin
    if(!rst_n) begin
        data_count <= 9'b0;
        frame_error <= 1'b0;
    end
    else begin
        if(frame_start) begin
            data_count <= 9'b0;
            frame_error <= 1'b0;
        end
        else if(fifo_wr_en && data_count < 9'd256) begin
            data_count <= data_count + 1'b1;
        end
        else if(frame_done) begin
            // Check if we collected all 256 data points
            if(data_count != 9'd256)
                frame_error <= 1'b1;
        end
    end
end

// State register
always @(posedge wr_clk or negedge rst_n) begin
    if(!rst_n)
        current_state <= WR_IDLE;
    else if(wr_rst_busy)
        current_state <= WR_IDLE;
    else
        current_state <= next_state;
end

// State transition logic
always @(*) begin
    case(current_state)
        WR_IDLE: begin
            if(!wr_rst_busy && frame_active)
                next_state = WR_WAIT_DATA;
            else
                next_state = WR_IDLE;
        end
        
        WR_WAIT_DATA: begin
            if(data_valid_d0 && !ad_otr_1)
                next_state = WR_CHECK;
            else if(frame_done)
                next_state = WR_IDLE;
            else
                next_state = WR_WAIT_DATA;
        end
        
        WR_CHECK: begin
            if(!full && !almost_full)
                next_state = WR_DATA;
            else
                next_state = WR_IDLE;
        end
        
        WR_DATA: begin
            if(full || almost_full || data_count == 9'd256)
                next_state = WR_IDLE;
            else
                next_state = WR_WAIT_DATA;
        end
        
        default: next_state = WR_IDLE;
    endcase
end

// Output logic
always @(posedge wr_clk or negedge rst_n) begin
    if(!rst_n) begin
        fifo_wr_en   <= 1'b0;
        fifo_wr_data <= 8'b0;
        data_valid   <= 1'b0;
        fifo_overflow <= 1'b0;
    end
    else begin
        // Overflow detection
        if(full && current_state == WR_CHECK)
            fifo_overflow <= 1'b1;
        else
            fifo_overflow <= 1'b0;
            
        case(current_state)
            WR_IDLE: begin
                fifo_wr_en   <= 1'b0;
                fifo_wr_data <= 8'b0;
                data_valid   <= 1'b0;
            end
            
            WR_WAIT_DATA: begin
                fifo_wr_en   <= 1'b0;
                data_valid   <= 1'b0;
            end
            
            WR_CHECK: begin
                fifo_wr_en   <= 1'b0;
                fifo_wr_data <= ad_data_sync;
                data_valid   <= 1'b1;
            end
            
            WR_DATA: begin
                fifo_wr_en   <= 1'b1;
                data_valid   <= 1'b1;
            end
            
            default: begin
                fifo_wr_en   <= 1'b0;
                fifo_wr_data <= 8'b0;
                data_valid   <= 1'b0;
            end
        endcase
    end
end

endmodule

