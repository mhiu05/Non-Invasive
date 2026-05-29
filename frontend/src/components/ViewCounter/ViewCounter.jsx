import React, { useEffect, useState } from 'react';
import { supabase } from '../../lib/supabase';
import { Users } from 'lucide-react';
import './ViewCounter.css';

const ViewCounter = ({ pageSlug = 'home' }) => {
  const [views, setViews] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const incrementViews = async () => {
      try {
        // Gọi hàm RPC trên Supabase để tăng lượt view an toàn (Atomic increment)
        // Hàm này sẽ tự cộng 1 vào cột view_count và trả về số lượng mới
        const { data, error } = await supabase.rpc('increment_page_view', { page_slug: pageSlug });
        
        if (error) {
          throw error;
        }
        
        if (data !== null) {
           setViews(data);
        }
      } catch (error) {
        console.error("Lỗi khi đếm lượt truy cập:", error.message);
        
        // Fallback: nếu RPC chưa được tạo hoặc lỗi, thử đọc số lượt xem hiện tại (chỉ đọc)
        try {
            const { data: fallbackData } = await supabase
              .from('page_views')
              .select('view_count')
              .eq('slug', pageSlug)
              .single();
              
            if (fallbackData) {
              setViews(fallbackData.view_count);
            }
        } catch(e) {
          // Bỏ qua lỗi fallback
        }
      } finally {
        setLoading(false);
      }
    };

    // Chỉ gọi hàm khi chạy trên trình duyệt (tránh lỗi nếu có SSR)
    if (typeof window !== 'undefined') {
      incrementViews();
    }
  }, [pageSlug]);

  return (
    <div className="view-counter-badge" title="Tổng lượt truy cập">
      <Users size={16} className="view-icon" />
      <span className="view-text">
        <span className="view-number">
          {loading ? '...' : (views ? views.toLocaleString() : '1')}
        </span> 
        lượt truy cập
      </span>
    </div>
  );
};

export default ViewCounter;
